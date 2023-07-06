# Docker Setup
The full FFMPerative installation uses FFmpeg 6.0 to support more filters.

### Windows & Mac Setup
Docker is the best way to setup for Windows and Mac users.

#### Get the Docker Image
Pull an image from DockerHub:
```bash
docker pull smellslikeml/ffmperative:latest
```

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
