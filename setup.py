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
    version="0.0.6",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'ffmperative': ['bin/ffmp'],
    },
    install_requires=read_requirements((this_directory / 'requirements.txt')),
    entry_points={
        "console_scripts": [
            "ffmperative=ffmperative.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
