# Iron Man Hand UI - HUD Demo

A futuristic hand-tracking UI with Iron Man-style HUD overlay using MediaPipe and OpenCV.

## Features

- Real-time hand tracking with MediaPipe
- Iron Man-inspired HUD elements:
  - Multi-ring rotating circles
  - Dashed arcs and ticks
  - Orbiting markers
  - Target brackets
  - Hex ring formations
  - Fingertip modules with mini orbits
  - Convex hull hand outline
  - Glitch-style text overlays
- Gesture detection (fist to hide HUD)
- Smooth hand orientation tracking

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### With Webcam (Normal Mode)

```bash
python hand_ui_demo.py
```

The script will automatically try camera indices 0, 1, and 2 to find a working webcam.

### Demo Mode (No Camera Required)

If your camera isn't working or you want to test the HUD visuals:

```bash
python hand_ui_demo.py --demo
```

This generates synthetic hand movements so you can see the HUD in action.

## Controls

- **ESC** or **Q**: Exit
- **Open palm**: Show HUD
- **Make fist**: Hide HUD
- **Rotate hand**: Changes displayed text

## Troubleshooting

### Camera Not Working

1. **Close other apps** using the camera (Zoom, Teams, Skype, etc.)
2. **Check Windows Privacy Settings**:
   - Settings > Privacy > Camera
   - Enable "Allow apps to access your camera"
3. **Try an external USB webcam**
4. **Use demo mode** as shown above

### Installation Issues

If you get import errors:

```bash
pip install --upgrade opencv-python mediapipe numpy
```

## Requirements

- Python 3.8+
- Webcam (optional in demo mode)
- opencv-python >= 4.8.0
- mediapipe >= 0.10.12
- numpy >= 1.24.0

## How It Works

1. **Hand Detection**: MediaPipe detects 21 hand landmarks in real-time
2. **Orientation Tracking**: Calculates hand rotation using wrist-to-index vector
3. **HUD Rendering**: Draws animated overlays anchored to hand center
4. **Gesture Recognition**: Detects fist by measuring fingertip-to-palm distances
5. **Smooth Animation**: Exponential smoothing for stable tracking

Enjoy your Iron Man hand UI! 🤖✋
