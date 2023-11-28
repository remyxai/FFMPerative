from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def read_requirements(file):
    with open(file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ffmperative",
    version="0.0.5-1",
    packages=find_packages(),
    install_requires=read_requirements((this_directory / 'requirements.txt')),
    extras_require={
        'full':  read_requirements((this_directory / 'requirements_full.txt'))
    },
    entry_points={
        "console_scripts": [
            "ffmp=ffmperative.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
