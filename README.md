# FFMPerative
## Devilishly Simple Video Processing

Large Language Models (LLMs) with Tools can perform complex tasks from natural language prompts. Based on HuggingFace's Agents & Tools, our agent is equipped with a suite of tools for common video processing workflows like:

* Get Video Metadata
* Extract Frame at Frame Number
* Make a Video from a Directory of Images 
* Horizontal/Vertical Flip
* Crop Video to Bounding Box
* Speed Up Video by X-factor
* Compress a GIF/Video

## Install
Clone this repo and install using pip
```
git clone https://github.com/remyxai/FFMPerative.git 
cd FFMPerative/
pip install .
```

### Quickstart
Getting started is easy, import the library and call the ffmp function.
```python

from ffmperative import ffmp

ffmp(prompt="crop video '/path/to/video.mp4' to 200,200,400,400 before writing to '/path/to/video_cropped.mp4', then double the speed of that video and write to '/path/to/video_cropped_fast.mp4'")
```

### CLI
You can also call FFMperative from the command line, try:
```bash
ffmp chat --prompt="sample the 5th frame from /path/to/video.mp4"
```

### Roadmap

- [x] Basic Video Tools
- [ ] Release LLM checkpoint fine-tuned to use ffmp Tools


### Contributing

* We'll gladly review pull requests aimed at improving the library of simple image and video processing tools.
* Interested in contributing to data/templates for specializing an LLM for video processing workflows, ping us!

Resources:
* [Huggingface Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
