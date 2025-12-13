import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import sys
from collections import deque
# -----------------------
# Config
# -----------------------
FRAME_W, FRAME_H = 1280, 720
SMOOTH_ALPHA = 0.7   # exponential smoothing for angle/pos
FIST_DIST_THRESHOLD = 0.10  # relative to hand box diagonal
TEXT_POOL = [
    "the Fut_ure is IN OUR HANDS",
    "ACCESS GRANTED",
    "SYSTEM ONLINE",
    "VECTOR LOCK",
    "RECALIBRATE",
    "SYNC"
]

# visual style
COLOR = (255, 255, 255)
THICK = 2
GLOW = 0.9
RNG = np.random.default_rng(1234)
# -----------------------
mp_hands = mp.solutions.hands
try:
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
except Exception as e:
    print("[ERROR] Failed to initialize MediaPipe Hands:", e)
    print("Hint: pip install mediapipe; ensure CPU/GPU drivers are okay.")
    raise
mp_draw = mp.solutions.drawing_utils

def normalized_to_px(norm_landmark, w, h):
    return int(norm_landmark.x * w), int(norm_landmark.y * h)

def landmarks_to_np(lm_list, w, h):
    pts = [normalized_to_px(lm, w, h) for lm in lm_list]
    return np.array(pts)

def hand_center(pts):
    return tuple(np.mean(pts, axis=0).astype(int))

def angle_from_vector(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    ang = math.degrees(math.atan2(dy, dx))  # -180 .. 180
    return ang

def rotate_points(points, angle_deg, origin):
    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    pts = np.array(points) - origin
    rotated = (R @ pts.T).T + origin
    return rotated.astype(int)

def _ellipse(img, center, r, a0, a1, color=COLOR, thickness=THICK):
    cv2.ellipse(img, center, (r, r), 0, a0, a1, color, thickness, cv2.LINE_AA)

def draw_dashed_arc(img, center, r, a0, a1, dash_deg=7, gap_deg=7, 
                    offset=0, thickness=THICK, color=COLOR):
    a0 += offset
    a1 += offset
    if a1 < a0:
        a0, a1 = a1, a0
    a = a0
    step = dash_deg + gap_deg
    while a < a1:
        seg_end = min(a + dash_deg, a1)
        cv2.ellipse(img, center, (r, r), 0, a, seg_end, color, thickness, cv2.LINE_AA)
        a += step

def draw_ticks(img, center, r, every_deg=10, len_px=10, angle_offset=0, color=COLOR):
    cx, cy = center
    for a in range(0, 360, every_deg):
        ang = math.radians(a + angle_offset)
        x0 = int(cx + r * math.cos(ang))
        y0 = int(cy + r * math.sin(ang))
        x1 = int(cx + (r - len_px) * math.cos(ang))
        y1 = int(cy + (r - len_px) * math.sin(ang))
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

def draw_orbiters(img, center, r, t, count=3, speeds=(40, 70, 110), color=COLOR):
    cx, cy = center
    for i in range(count):
        ang = (t * speeds[i % len(speeds)]) % 360
        rad = math.radians(ang + i * 120)
        x = int(cx + r * math.cos(rad))
        y = int(cy + r * math.sin(rad))
        cv2.circle(img, (x, y), max(2, r // 18), color, -1, cv2.LINE_AA)

def draw_glyphs(img, center, r, angle_deg):
    cx, cy = center
    pts = [
        (cx + int(r * 0.35), cy - int(r * 0.05)),
        (cx + int(r * 0.25), cy + int(r * 0.22)),
        (cx - int(r * 0.20), cy - int(r * 0.28)),
    ]
    pts = rotate_points(pts, angle_deg, np.array([cx, cy]))
    for p in pts:
        cv2.circle(img, tuple(p), max(3, r // 14), COLOR, 1, cv2.LINE_AA)
        cv2.line(img, center, tuple(p), COLOR, 1, cv2.LINE_AA)

def draw_target_brackets(img, center, r, angle=0, size_ratio=0.22):
    cx, cy = center
    s = max(12, int(r * size_ratio))
    d = int(r * 1.25)
    pts = [
        (cx - d, cy - d), (cx + d, cy - d), (cx + d, cy + d), (cx - d, cy + d)
    ]
    pts = rotate_points(pts, angle, np.array([cx, cy]))
    for (x, y) in pts:
        cv2.line(img, (x - s, y), (x + s//2, y), COLOR, 2, cv2.LINE_AA)
        cv2.line(img, (x, y - s), (x, y + s//2), COLOR, 2, cv2.LINE_AA)

def _hex_points(center, r):
    cx, cy = center
    pts = []
    for a in range(0, 360, 60):
        rad = math.radians(a)
        pts.append((int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))))
    return np.array(pts, dtype=np.int32)

def draw_hex_ring(img, center, base_r, count=8, hex_r=10, offset=0):
    cx, cy = center
    for i in range(count):
        a = math.radians(offset + i * (360.0 / count))
        x = int(cx + base_r * math.cos(a))
        y = int(cy + base_r * math.sin(a))
        poly = _hex_points((x, y), hex_r)
        cv2.polylines(img, [poly], True, COLOR, 1, cv2.LINE_AA)

def draw_fingertip_modules(img, pts, angle, t, base_r):
    tip_indices = [4, 8, 12, 16, 20]
    for i, idx in enumerate(tip_indices):
        x, y = pts[idx]
        ring = int(18 + 6 * math.sin(t * (2 + i)))
        cv2.circle(img, (x, y), ring, COLOR, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(2, ring // 4), COLOR, -1, cv2.LINE_AA)
        # tiny rotating dash
        ang = math.radians((t * 180 + i * 45) % 360)
        px = int(x + (ring + 6) * math.cos(ang))
        py = int(y + (ring + 6) * math.sin(ang))
        cv2.circle(img, (px, py), 2, COLOR, -1, cv2.LINE_AA)
        # line to hub
        cv2.line(img, (x, y), (x + (pts[0][0]-x)//6, y + (pts[0][1]-y)//6), COLOR, 1, cv2.LINE_AA)

def draw_hand_hull(img, pts):
    hull = cv2.convexHull(pts)
    cv2.polylines(img, [hull], True, (255, 255, 255), 1, cv2.LINE_AA)

def draw_glitch_text(img, text, anchor, t):
    # simple HUD-like glitch: layered text with jitter, small fragments
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    base = (255, 255, 255)
    x, y = anchor
    jitter = int(2 + 2 * math.sin(t * 7))
    # main
    cv2.putText(img, text, (x, y), font, scale, base, 2, cv2.LINE_AA)
    # ghost layers
    cv2.putText(img, text, (x + 1, y - 1), font, scale, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x - 1, y + 1), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    # small broken pieces
    if RNG.random() < 0.35:
        w = cv2.getTextSize(text, font, scale, 1)[0][0]
        sx = x + int(RNG.random() * w)
        ex = sx + int(RNG.random() * 30 + 10)
        yy = y - 10 + int(RNG.random() * 20) - 10
        cv2.line(img, (sx, yy), (ex, yy), (255, 255, 255), 1, cv2.LINE_AA)

def draw_tech_circle(overlay, center, radius, angle_deg, text, t):
    cx, cy = center
    # animation offsets
    spin_fast = (t * 120) % 360
    spin_slow = (t * 40) % 360
    angle = angle_deg

    # outer rings and dashes
    _ellipse(overlay, center, radius, 0 + angle, 360 + angle, COLOR, 2)
    draw_dashed_arc(overlay, center, int(radius * 0.88), 0, 360, 8, 5, offset=spin_slow + angle, thickness=2)
    draw_dashed_arc(overlay, center, int(radius * 1.08), 20, 340, 20, 8, offset=-spin_fast + angle, thickness=1)

    # inner core
    _ellipse(overlay, center, int(radius * 0.55), -30 + angle, 210 + angle, COLOR, 2)
    _ellipse(overlay, center, int(radius * 0.34), 0, 360, COLOR, 1)
    cv2.circle(overlay, center, max(6, radius // 10), COLOR, 1, cv2.LINE_AA)
    draw_ticks(overlay, center, int(radius * 0.72), every_deg=12, len_px=int(radius * 0.06), angle_offset=spin_fast + angle)

    # spokes
    spokes = [
        (cx + int(radius * 0.75), cy),
        (cx + int(radius * 0.55), cy - int(radius * 0.18)),
        (cx + int(radius * 0.55), cy + int(radius * 0.18))
    ]
    for p in rotate_points(spokes, angle, np.array([cx, cy])):
        cv2.line(overlay, center, tuple(p), COLOR, 1, cv2.LINE_AA)

    # orbiters and glyphs
    draw_orbiters(overlay, center, int(radius * 1.2), t, count=3)
    draw_glyphs(overlay, center, radius, angle)
    # target brackets and hex ring
    draw_target_brackets(overlay, center, radius, angle)
    draw_hex_ring(overlay, center, int(radius * 1.45), count=10, hex_r=max(6, radius // 10), offset=spin_slow + angle)

    # text block (to the left/lower of the HUD)
    txt_anchor = (cx - radius, cy + radius + 30)
    draw_glitch_text(overlay, text, txt_anchor, t)

# smoothing state
smooth_angle = None
smooth_center = None
current_text = random.choice(TEXT_POOL)
last_text_angle = None

# Check for demo mode
DEMO_MODE = '--demo' in sys.argv

if not DEMO_MODE:
    # Try multiple camera indices
    working_cam_idx = None
    for cam_idx in [0, 1, 2]:
        print(f"[INFO] Trying camera index {cam_idx}...")
        test_cap = cv2.VideoCapture(cam_idx)
        if test_cap.isOpened():
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                print(f"[SUCCESS] Camera {cam_idx} is working!")
                working_cam_idx = cam_idx
                test_cap.release()
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if working_cam_idx is None:
        print("\n[ERROR] No working camera found!")
        print("Solutions:")
        print("  1. Close other apps using the camera (Zoom, Teams, etc.)")
        print("  2. Check Windows Privacy Settings > Camera > Allow apps to access camera")
        print("  3. Try an external USB webcam")
        print("  4. Run in demo mode: python hand_ui_demo.py --demo")
        raise SystemExit(1)
    
    # Create fresh capture with the working index
    cap = cv2.VideoCapture(working_cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print(f"[INFO] Camera initialized: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
else:
    print("[INFO] Running in DEMO mode (no camera required)")
    cap = None

# Demo mode: generate synthetic frames and hand landmarks
def generate_demo_hand(t, w, h):
    """Generate synthetic hand landmarks for demo mode"""
    cx = int(w // 2 + 150 * math.sin(t * 0.5))
    cy = int(h // 2 + 100 * math.cos(t * 0.3))
    angle = t * 30 % 360
    
    # Create 21 landmarks in hand shape
    landmarks = []
    # Wrist
    landmarks.append(type('obj', (object,), {'x': cx/w, 'y': cy/h, 'z': 0})())
    
    # Thumb (4 points)
    for i in range(1, 5):
        offset_x = (cx + 30 + i * 15) / w
        offset_y = (cy - 20 - i * 10) / h
        landmarks.append(type('obj', (object,), {'x': offset_x, 'y': offset_y, 'z': 0})())
    
    # Index to pinky (4 fingers x 4 points = 16)
    for finger in range(4):
        base_x = cx - 40 + finger * 30
        for i in range(1, 5):
            offset_x = base_x / w
            offset_y = (cy - 50 - i * 20) / h
            landmarks.append(type('obj', (object,), {'x': offset_x, 'y': offset_y, 'z': 0})())
    
    return type('obj', (object,), {'landmark': landmarks})()

frame_count = 0
while True:
    if DEMO_MODE:
        # Generate synthetic frame
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        t = (time.time() % 1000.0)
        
        # Create fake detection result
        demo_hand = generate_demo_hand(t, w, h)
        res = type('obj', (object,), {'multi_hand_landmarks': [demo_hand]})()
    else:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read a frame from camera.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t = (time.time() % 1000.0)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
    
    frame_count += 1

    overlay = np.zeros_like(frame)  # draw overlay here, then blend
    draw_overlay = False
    fist_detected = False

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        pts = landmarks_to_np(hand.landmark, w, h)
        # hand bounding box diag (for normalization)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        diag = math.hypot(x_max-x_min, y_max-y_min) if (x_max-x_min)>0 else 1

        # pick orientation vector: wrist (0) -> index_mcp (5)
        p_wrist = tuple(pts[0])
        p_index = tuple(pts[5])
        angle = angle_from_vector(p_wrist, p_index)  # -180..180

        # smooth angle and center
        center = hand_center(pts)
        if smooth_angle is None:
            smooth_angle = angle
            smooth_center = center
        else:
            # angular smoothing: shortest path
            diff = (angle - smooth_angle + 180) % 360 - 180
            smooth_angle = smooth_angle + diff * (1 - SMOOTH_ALPHA)
            smooth_center = (int(smooth_center[0] * SMOOTH_ALPHA + center[0] * (1-SMOOTH_ALPHA)),
                             int(smooth_center[1] * SMOOTH_ALPHA + center[1] * (1-SMOOTH_ALPHA)))

        # fist detection: average tip-to-palm distance
        # palm center: use landmark 0..4..9 average (approx)
        palm_center = np.mean(pts[[0,1,2,5,9]], axis=0)
        tip_indices = [4, 8, 12, 16, 20]
        dists = [math.hypot(pts[i][0]-palm_center[0], pts[i][1]-palm_center[1]) for i in tip_indices]
        mean_dist = sum(dists)/len(dists)
        rel = mean_dist / diag  # normalized
        if rel < FIST_DIST_THRESHOLD:
            fist_detected = True
        else:
            fist_detected = False

        # rotation-triggered random text: if angle change big enough, change text
        if last_text_angle is None:
            last_text_angle = smooth_angle
        if abs(((smooth_angle - last_text_angle + 180) % 360) - 180) > 20:
            current_text = random.choice(TEXT_POOL)
            last_text_angle = smooth_angle

        # compute radius scaled from bbox
        radius = max(40, int(diag * 0.28))

        if not fist_detected:
            draw_overlay = True
            draw_tech_circle(overlay, smooth_center, radius, smooth_angle, current_text, t)
            # fingertip mini modules + hull
            draw_fingertip_modules(overlay, pts, smooth_angle, t, radius)
            draw_hand_hull(overlay, pts)

        # Optional: draw landmarks for debug
        # mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # blend overlay (white lines) as semi-transparent
    if draw_overlay:
        # white-on-black overlay -> convert white to color then blend
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
        colored = overlay.copy()
        # make the overlay glow: change thickness or gaussian blur for glow
        blurred = cv2.GaussianBlur(colored, (11,11), 0)
        # composite: where mask, use overlay; else frame
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        fg = cv2.bitwise_and(blurred, blurred, mask=mask)
        frame = cv2.addWeighted(bg, 1.0, fg, GLOW, 0)
    else:
        # optionally fade out text if fist
        pass

    cv2.imshow("Hand UI - Iron Man HUD" + (" [DEMO MODE]" if DEMO_MODE else ""), frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited cleanly")
