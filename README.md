# FFMPerative
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
</p>

## Video Production at the Speed of Chat

Large Language Models (LLMs) with Tools can tackle tough tasks framed in simple language. Consider FFMPerative your copilot for common video processing workflows such as:

* Get Video Metadata
* Sample Image from Video
* Change Video Playback Speed
* Make a Video from a Directory of Images 
* Resize, Crop, Flip, Reverse Video/GIF
* Adjust Audio Levels, Background Noise Removal
* Overlay Video for Picture-in-Picture
* Split Video with Scene Detection

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

With more compositition, you can even curate highlights from long-form video by analyzing speech transcripts [with LLMs](https://blog.remyx.ai/posts/data-processing-agents/):

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
Import the library and pass your prompt to `ffmp`.
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
