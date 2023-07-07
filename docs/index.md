# FFMPerative Examples

## Basic Usage
Probe for metadata info about your source media:
```
ffmperative "get info from 'hotel_lobby.mp4'"
```

Chunk your video into 3 second GOPs:
```
ffmperative "chunk 'video.mp4' into 3 second clips"
```

Pad portrait mode videos with letterboxing using the python CLI:
```
ffmp do --p "apply letterboxing to 'video.mp4' call it 'video_letterbox.mp4'"
```

Vertically stack two videos:
```
ffmp do --p "vertically stack 'video1.mp4' and 'video2.mp4' calling it 'result.mp4'"
```

## Advanced Usage

Apply VidStab to stabilize your video:
```
ffmperative "stabilize 'video.mp4'"
```

Apply Ken Burns effect to zoompan an image to video:
```
ffmp("ken burns effect on 'image.png' call it 'image_kenburns.mp4'")
```

Perform speech-to-text transcription on your video:
```
ffmperative "get subtitles from 'video.mp4'"
```

Apply an image classifier to every 5th frame from your video:
```
ffmp do --p "classify 'video.mp4' using the model 'my_model/my_model.onnx'"
```

