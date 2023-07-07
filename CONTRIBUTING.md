# CONTRIBUTING

* Have a video processing workflow in mind? Want to contribute to our project? We'd love to hear from you! Raise an issue and we'll work together to design it.

### Building a Wheel
```
cd FFMPerative
python3 -m build --wheel
```
Now you can install the wheel with pip:
```
pip install dist/ffmperative-X.X.X-py3-none-any.whl
```

### Building a Docker Image & Running a Container
Or clone this repo and build an image with the `Dockerfile`:
```bash
git clone https://github.com/remyxai/FFMPerative.git
cd FFMPerative/docker
docker build -t ffmperative .
```

#### Run FFMPerative in a Container
```bash
docker run -it -e HUGGINGFACE_TOKEN='YOUR_HF_TOKEN' -v /path/to/dir:/path/to/dir --entrypoint /bin/bash ffmperative:latest
```

### Build a Debian Package from Source
For debian, build and install the package:
```bash
dpkg-deb --build package_build/ ffmperative.deb
sudo dpkg -i ffmperative.deb
```

