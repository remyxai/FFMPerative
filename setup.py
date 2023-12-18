import os
import requests
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

        # Download the model
        self.download_model(bin_dir)

        # Call the standard install command
        install.run(self)

    @staticmethod
    def download_model(bin_dir):
        model_url = f"https://remyx.ai/assets/ffmperative/{version}/ffmp"
        target_path = os.path.join(bin_dir, 'ffmp')

        if not os.path.exists(target_path):
            print("Downloading model assets...")
            response = requests.get(model_url, stream=True)
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        return target_path

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
