# FFMPerative
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
</p>

## Video Production at the Speed of Chat

Large Language Models (LLMs) with Tools can perform complex tasks from natural language prompts. Using HuggingFace's Agents & Tools, FFMPerative is specialized to common video processing workflows like:

* Get Video Metadata
* Sample Image from Video
* Change Video Playback Speed
* Make a Video from a Directory of Images 
* Resize, Crop, Flip, Reverse Video/GIF
* Adjust Audio Levels, Background Noise Removal
* Overlay Video for Picture-in-Picture
* Split Video with Scene Detection

## Setup 

### Installation
For Linux users, simply install via aptitude:
```
dpkg-deb --build package_build/ ffmperative.deb
sudo dpkg -i ffmperative.deb
```
Then run the following line to configure the package to mount the directory at `/home/$(hostname)/Videos/`:
```
echo -e "HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN\nVIDEOS_PATH=/home/$(hostname)/Videos" | sudo tee /etc/ffmperative/config
```



### Windows & Mac
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

## Quickstart

Editing video with FFMPerative is simple.

Run FFMPerative from the command-line:
```bash
ffmperative "sample the 5th frame from /path/to/video.mp4"
```

As above, you can sample a frame from a video clip. You can also split a long video into short clips via scene detection:

```bash
ffmperative "split the video '/path/to/my_video.mp4' by scene"
```

Another common workflow, adding closed-captioning:

```bash
ffmperative "merge subtitles '/path/to/captions.srt' with video '/path/to/my_video.mp4' calling it '/path/to/my_video_captioned.mp4'"
```

By compositition, you can even curate highlights from long-form video by analyzing speech transcripts [with LLMs](https://blog.remyx.ai/posts/data-processing-agents/):

![smart_trim](https://blog.remyx.ai/img/ffmperative-auto-edit-pipeline.png#center)

## Features

### Python Bindings
Import the library and pass your prompt as argument to `ffmp`.
```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

### Notebooks

* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Roadmap

- [x] Basic Video Tools
- [x] Release to PyPI after Agents are Added to Transformers
- [x] Add ML-enabled Tools: [demucs](https://github.com/facebookresearch/demucs), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) 
- [x] Docker Image with Latest FFmpeg
- [ ] Host .deb package for `apt-get` installation
- [ ] Release LLM checkpoint fine-tuned to use ffmp Tools


### Contributing

* Have a video processing workflow in mind? Raise an issue and we'll try helping to design it!

### Resources
* [Huggingface Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)

### Community

* [B-Roll](https://b-roll.ai/)
* [@brollai](https://twitter.com/brollai)
