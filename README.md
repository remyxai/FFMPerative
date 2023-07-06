# FFMPerative
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
  <img src="https://img.shields.io/pypi/v/ffmperative.svg">
  <img src="https://img.shields.io/github/license/remyxai/ffmperative.svg">
  <img src="https://img.shields.io/docker/v/smellslikeml/ffmp">
</p>

## Video Production at the Speed of Chat

FFMPerative is your copilot for video production workflows. Powered by Large Language Models (LLMs) and an intuitive chat interface, it makes complex tasks as simple as typing a sentence. Leverage the power of FFmpeg and cutting-edge machine learning tools without dealing with complex command-line arguments or scripts.

* Get Video Metadata
* Sample Image from Video
* Change Video Playback Speed
* Apply FFmpeg xfade transition filters
* Resize, Crop, Flip, Reverse Video/GIF
* Make a Video from a Directory of Images 
* Overlay Image & Video for Picture-in-Picture
* Adjust Audio Levels, Background Noise Removal
* Speech-to-Text Transcription and Closed-Captions
* Split Video by N-second Gops or with Scene Detection
* Image Classifier Inference on every N-th Video Frame

## Setup 

### Debian Package
For debian, build and install the package:
```
dpkg-deb --build package_build/ ffmperative.deb
sudo dpkg -i ffmperative.deb
```
Configure the package to mount the directory at `/home/$(hostname)/Videos/` by running:
```
echo -e "HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN\nVIDEOS_PATH=/home/$(hostname)/Videos" | sudo tee /etc/ffmperative/config
```

## Quickstart

To sample an image from a video clip, simply run FFMPerative from the command-line:

```bash
ffmperative "sample the 5th frame from /path/to/video.mp4"
```

Similarly, it's simple to split a long video into short clips via scene detection:

```bash
ffmperative "split the video '/path/to/my_video.mp4' by scene"
```

Try adding closed-captions with:

```bash
ffmperative "merge subtitles '/path/to/captions.srt' with video '/path/to/my_video.mp4' calling it '/path/to/my_video_captioned.mp4'"
```

FFMPerative excels in task compositition. For instance, [curate video highlights](https://blog.remyx.ai/posts/data-processing-agents/) by analyzing speech transcripts:

![smart_trim](https://blog.remyx.ai/img/ffmperative-auto-edit-pipeline.png#center)


### Windows & Mac Setup
#### Get the Docker Image
Pull an image from DockerHub:
```
docker pull smellslikeml/ffmp:latest
```

Or clone this repo and build an image with the `Dockerfile`:
```
git clone https://github.com/remyxai/FFMPerative.git
cd FFMPerative
docker build -t ffmp .
```

#### Run FFMPerative in a Container
```
docker run -it -e HUGGINGFACE_TOKEN='YOUR_HF_TOKEN' -v /path/to/dir:/path/to/dir --entrypoint /bin/bash ffmp:latest
```


## Features

### Python Usage

You can also use FFMPerative in your Python projects. Simply import the library and pass your command as a string to `ffmp`.

```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

### Notebooks

Explore our notebooks for practical applications of FFMPerative:

* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Roadmap

- [x] Basic Video Tools
- [x] Release to PyPI after Agents are Added to Transformers
- [x] Add ML-enabled Tools: [demucs](https://github.com/facebookresearch/demucs), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) 
- [x] Docker Image with Latest FFmpeg
- [ ] Host .deb package for `apt-get` installation
- [ ] Release LLM checkpoint fine-tuned to use ffmp Tools


### Contributing

* Have a video processing workflow in mind? Want to contribute to our project? We'd love to hear from you! Raise an issue and we'll work together to design it.


### Resources
* [Huggingface Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)

### Community

* [B-Roll](https://b-roll.ai/)
* [@brollai](https://twitter.com/brollai)
