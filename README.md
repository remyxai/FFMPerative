# FFMPerative
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/mascot.png" height=200px>
</p>

## Devilishly Simple Video Processing

Large Language Models (LLMs) with Tools can perform complex tasks from natural language prompts. Using HuggingFace's Agents & Tools, FFMPerative is specialized to common video processing workflows like:

* Get Video Metadata
* Sample Image from Video
* Change Video Playback Speed
* Make a Video from a Directory of Images 
* Resize, Flip, Crop, Compress Video/GIF
* Adjust Audio Levels, Background Noise Removal
* Overlay Video for Picture-in-Picture
* Split Video with Scene Detection

## Prerequisites 

Some tools require additional options of a [special build](https://johnvansickle.com/ffmpeg/) of FFmpeg. 

```bash
mkdir ffmpeg
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar xf ffmpeg-git-amd64-static.tar.xz -C ffmpeg --strip-components 1
```

Make sure to update your PATH variable:

`export PATH=$PATH:/path/to/ffmpeg`


## Installation

#### PyPI
Install from pypi with:
```
pip install ffmperative
```

#### From Source
Clone this repo and install using pip
```
git clone https://github.com/remyxai/FFMPerative.git 
cd FFMPerative/
pip install .
```

## Quickstart

Getting started is easy, import ffmp from the library and specify your edit in simple terms.

```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

Besides sampling a frame from a clip, we can split a long video into short clips with scene detection:

```python
ffmp("split the video '/path/to/my_video.mp4' by scene")
```

Another common workflow, adding closed-captioning:

```python
ffmp("merge subtitles '/path/to/captions.srt' with video '/path/to/my_video.mp4' calling it '/path/to/my_video_captioned.mp4'")
```

With more compositition, you can even curate highlights from long-form video by analyzing Speech-To-Text transcriptions [with LLMs](https://blog.remyx.ai/posts/data-processing-agents/):

![smart_trim](https://blog.remyx.ai/img/ffmperative-auto-edit-pipeline.png#center)

## Features

### CLI
Run FFMPerative from the command-line:
```bash
ffmp do -p "sample the 5th frame from /path/to/video.mp4"
```

### Notebooks

* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Roadmap

- [x] Basic Video Tools
- [x] Release to PyPI after Agents are Added to Transformers
- [x] Add ML-enabled Tools: [demucs](https://github.com/facebookresearch/demucs), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) 
- [ ] Release LLM checkpoint fine-tuned to use ffmp Tools


### Contributing

* Have a video processing workflow in mind? Raise an issue and we'll try helping to design it!

### Resources
* [Huggingface Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)

### Community

* [B-Roll](https://b-roll.ai/)
* [@brollai](https://twitter.com/brollai)
