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

See [Docker Setup](docker/README.md).
You can pull the prebuilt image [smellslikeml/ffmperative:0.0.6-min](https://hub.docker.com/layers/smellslikeml/ffmperative/0.0.6-min/images/sha256-833489f673f7f2153d4c59b2fcfdd54baf181533c8196a3abedcf4d362bfddc2?context=repo)

## Quickstart
To sample an image from a video clip, simply run FFMPerative from the command-line:

```bash
ffmp do --prompt "sample the 5th frame from /path/to/video.mp4"
```

Similarly, it's simple to split a long video into short clips via scene detection:

```bash
ffmp do --prompt "split the video '/path/to/my_video.mp4' by scene"
```

Or to add closed-captions with:

```bash
ffmp do --prompt "merge subtitles 'captions.srt' with video 'video.mp4' calling it 'video_caps.mp4'"
```

FFMPerative excels in task compositition. For instance, [curate video highlights](https://blog.remyx.ai/posts/data-processing-agents/) by analyzing speech transcripts:

![smart_trim](https://blog.remyx.ai/img/ffmperative-auto-edit-pipeline.png#center)


## Features

### Python Usage
With `ffmpeg` installed on your system, you can opt for the minimal installation of FFMPerative through pip.

#### Setup
Make the minimal install of ffmperative with:

```bash
pip install git+https://github.com/remyxai/FFMPerative.git@minimal_pkg
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

### Compose 🎞️ 
Use the `compose` call to compose clips into an edited video. Use the optional `--prompt` flag to guide the composition by text prompt.
```bash
ffmp compose --clips /path/to/video/dir --output /path/to/my_video.mp4 --prompt "Edit the video for social media"
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
