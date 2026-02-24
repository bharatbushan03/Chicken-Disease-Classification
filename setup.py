import setuptools

with open("README.md", 'r', encoding = 'utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Chicken-Disease-Classification"
AUTHOR_USER_NAME = "your_username"  # Replace with your GitHub username
SRC_REPO = "CNNClassifier"
AUTHOR_EMAIL = "your_email@example.com"  # Replace with your email

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN-based chicken disease classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "tensorflow",
        "pandas",
        "dvc",
        "notebook",
        "numpy",
        "matplotlib",
        "seaborn",
        "python-box==6.0.2",
        "pyYAML",
        "tqdm",
        "ensure==1.0.2",
        "joblib",
        "types-pyYAML",
        "scipy",
        "Flask",
        "Flask-Cors"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)