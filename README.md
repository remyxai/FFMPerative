# FFMPerative - Chat to Compose Video
<p align="center">
  <img src="https://github.com/remyxai/FFMPerative/blob/main/assets/ffmperative.gif" height=400px>
  <br>
  <img src="https://img.shields.io/pypi/v/ffmperative.svg">
  <img src="https://img.shields.io/pypi/dm/ffmperative">
  <img src="https://img.shields.io/github/license/remyxai/ffmperative.svg">

</p>

FFMPerative is your copilot for video editing workflows. Powered by Large Language Models (LLMs) through an intuitive chat interface, now you can compose video edits in natural language to do things like:

* Change Speed, Resize, Crop, Flip, Reverse Video/GIF
* Speech-to-Text Transcription and Closed-Captions

Just describe your changes like [these examples](https://remyxai.github.io/FFMPerative/).

## Setup 

### Requirements
* Python 3 
* [ffmpeg](https://ffmpeg.org)

PyPI:
```
pip install ffmperative
```

Or pip install from source:
```
git clone https://github.com/remyxai/FFMPerative.git
cd FFMPerative && pip install .
```

## Quickstart
Add closed-captions with:

```bash
ffmperative do --prompt "merge subtitles 'captions.srt' with video 'video.mp4' calling it 'video_caps.mp4'"
```

## Features

### Python Usage
Simply import the library and pass your command as a string to `ffmp`.

```python
from ffmperative import ffmp

ffmp("sample the 5th frame from '/path/to/video.mp4'")
```

### Compose 🎞️ 
Use the `compose` call to compose clips into an edited video. Use the optional `--prompt` flag to guide the composition by text prompt.
```bash
ffmperative compose --clips /path/to/video/dir --output /path/to/my_video.mp4 --prompt "Edit the video for social media"
```

### Resources
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/)
* [Sample FFMPerative Dataset](https://huggingface.co/datasets/remyxai/ffmperative-sample)
* [FFMPerative LLaMA2 checkpoint](https://huggingface.co/remyxai/ffmperative-7b)
* [Automatically Edit Videos from Google Drive in Colab](https://colab.research.google.com/drive/149byzCNd17dAehVuWXkiFQ2mVe_icLCa?usp=sharing)

### Community
* [Join us on Discord](https://discord.com/invite/b2yGuCNpuC)

### Edit-Plan Verification — adapted from [Aurora: Unified Video Editing with a Tool-Using Agent](https://arxiv.org/abs/2605.18748)

Aurora pairs a tool-using VLM agent with a video generator and shows that
mapping a raw user request into a *structured, complete edit plan* — resolving
underspecification **before** generation — improves instruction-following.
FFMPerative adopts that idea at the point where the agent's generated tool
sequence is about to run: `ffmperative/edit_plan_check.py` parses the plan into
structured steps and verifies each one against the available tools, surfacing
**unknown tools** and **missing required arguments** before any `ffmpeg`
primitive executes. The check is wired into `interpretor.evaluate`, so every
plan is validated on the normal execution path. The diffusion/generation half
of Aurora is out of scope — FFMPerative composes `ffmpeg` primitives, so the
value is plan completeness over those primitives.

Contributed via [Remyx Recommendation](https://engine.remyx.ai).
