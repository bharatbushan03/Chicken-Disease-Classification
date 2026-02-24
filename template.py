import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: [%(message)s]')

project_name = 'CNNClassifier'

list_of_files = [
    '.github/workflows/.gitkeep',
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    'params.yaml',
    'requirements.txt',
    'setup.py',
<<<<<<< HEAD
    'research/trials.ipynb',
    "templates/index.html"
=======
    'research/trials.ipynb'
>>>>>>> 6baf30bebe597a1595b34f506f2d1c752d44e7de
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating file: {filepath}")

    else:
        logging.info(f"File already exists: {filepath}")