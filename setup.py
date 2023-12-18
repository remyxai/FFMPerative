import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

version = "0.0.7"

def read_requirements(file):
    with open(file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

class CustomInstall(install):
    def run(self):
        # Ensuring the bin/ directory exists
        bin_dir = os.path.join(self.install_lib, 'ffmperative', 'bin')
        os.makedirs(bin_dir, exist_ok=True)

        # Call the standard install command
        install.run(self)


setup(
    name="ffmperative",
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements((this_directory / 'requirements.txt')),
    entry_points={
        "console_scripts": [
            "ffmperative=ffmperative.cli:main",
        ],
    },
    cmdclass={
        'install': CustomInstall,
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
