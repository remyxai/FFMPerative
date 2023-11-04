# FFMPerative
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
  <br>
  <img src="https://img.shields.io/pypi/v/ffmperative.svg">
  <img src="https://img.shields.io/pypi/dm/ffmperative">
  <img src="https://img.shields.io/github/license/remyxai/ffmperative.svg">
  <img src="https://img.shields.io/docker/v/smellslikeml/ffmperative/latest">
  <img src="https://img.shields.io/docker/pulls/smellslikeml/ffmperative">

</p>

## Video Production at the Speed of Chat
FFMPerative is your copilot for video editing workflows. Powered by Large Language Models (LLMs) through an intuitive chat interface, now you can compose video edits in natural language. Integrate FFmpeg and cutting-edge machine learning tools without dealing with complex command-line arguments or scripts.

* Get Video Metadata
* Sample Image from Video
* Change Video Playback Speed
* Apply FFmpeg [xfade transition filters](https://trac.ffmpeg.org/wiki/Xfade#Gallery)
* Resize, Crop, Flip, Reverse Video/GIF
* Make a Video from a Directory of Images 
* Overlay Image & Video for Picture-in-Picture
* Adjust Audio Levels, Background Noise Removal
* Speech-to-Text Transcription and Closed-Captions
* Split Video by N-second Gops or with Scene Detection
* Image Classifier Inference on every N-th Video Frame

Just describe your desired edits similar to [these examples](https://remyxai.github.io/FFMPerative/).

## Setup 

### Ubuntu (Recommended)
Install the package via aptitude:

```bash
# add our PPA
sudo add-apt-repository ppa:remyxai/ppa
sudo apt update

# install
sudo apt-get install ffmperative
```

Configure with your huggingface token and your preferred video directory
```bash
ffmperative configure
```

For Windows & Mac, see [Docker Setup](docker/README.md).

## Quickstart
To sample an image from a video clip, simply run FFMPerative from the command-line:

```bash
ffmperative "sample the 5th frame from /path/to/video.mp4"
```

Similarly, it's simple to split a long video into short clips via scene detection:

```bash
ffmperative "split the video '/path/to/my_video.mp4' by scene"
```

Or to add closed-captions with:

```bash
ffmperative "merge subtitles 'captions.srt' with video 'video.mp4' calling it 'video_caps.mp4'"
```

FFMPerative excels in task compositition. For instance, [curate video highlights](https://blog.remyx.ai/posts/data-processing-agents/) by analyzing speech transcripts:

![smart_trim](https://blog.remyx.ai/img/ffmperative-auto-edit-pipeline.png#center)


## Features

### Python Usage
With `ffmpeg` installed on your system, you can opt for the minimal installation of FFMPerative through pip.

#### Setup
Make the minimal install of ffmperative with:

```bash
# from PyPI
pip install ffmperative
```

#### Usage
Simply import the library and pass your command as a string to `ffmp`.

```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

You can also use the command-line interface:
```bash
ffmp do --p "sample the 5th frame from '/path/to/video.mp4'"
```

### Notebooks

Explore our notebooks for practical applications of FFMPerative:

* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Resources
* [Huggingface Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [Sample FFMPerative Dataset](https://huggingface.co/datasets/remyxai/ffmperative-sample)
* [FFMPerative LLaMA2 checkpoint](https://huggingface.co/remyxai/ffmperative-7b)

### Community

* [B-Roll](https://b-roll.ai/)
* [@brollai](https://twitter.com/brollai)
