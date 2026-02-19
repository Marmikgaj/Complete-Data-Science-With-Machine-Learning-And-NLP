
from setuptools import setup, find_packages



# Method 1: 
# setup(
#     name="ml_project",
#     version="0.1.0",
#     author='Marmik Shah',
#     author_email='marmikgaj005@gmail.com',
#     packages=find_packages(),
#     install_requires=[
#         "numpy",
#         "pandas",
#         "seaborn",
#         "matplotlib",
#         "scikit-learn",
#     ],
# )

from typing import List
def get_requirments(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """

    requirements_list:List[str] = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        for requirement in requirements:
            requirement = requirement.replace("\n","")
            if requirement.startswith("#"):
                continue
            if requirement == '-e .':
                continue
            requirements_list.append(requirement)

    return requirements_list

# Method 2: 
setup(
    name="ml_project",
    version="0.1.0",
    author='Marmik Shah',
    author_email='marmikgaj005@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')
)