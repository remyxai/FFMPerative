import os
import re
import csv
import cv2
import math
import json
import ffmpeg
import numpy as np
import subprocess
from io import BytesIO
from PIL import Image
from pathlib import Path
from collections import Counter
from transformers import Tool

from .utils import modify_file_name, probe_video, get_video_info, has_audio

import whisperx
import demucs.separate
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import onnxruntime as rt


class AudioAdjustmentTool(Tool):
    name = "audio_adjustment_tool"
    description = """
    This tool modifies audio levels for an input video.
    Inputs are input_path, output_path, level (e.g. 0.5 or -13dB).
    Output is the output_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def __call__(self, input_path: str, output_path: str, level: str):
        (
            ffmpeg.input(input_path)
            .output(output_path, af="volume={}".format(level))
            .overwrite_output()
            .run()
        )
        return output_path


class AudioDemuxTool(Tool):
    name = "audio_demux_tool"
    description = """
    This tool performs music source separation to demux vocals
    from audio, good for kareoke or filtering background noise. 
    Inputs are input_path.
    """
    inputs = ["text"]
    outputs = ["None"]

    def __call__(self, input_path: str):
        demucs.separate.main(
            ["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", input_path]
        )


class AudioVideoMuxTool(Tool):
    name = "audio_video_mux_tool"
    description = """
    This tool muxes (combines) a video and an audio file.
    Inputs are input_path as a string, audio_path as a string, and output_path as a string.
    Output is the output_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, audio_path: str, output_path: str):
        input_video = ffmpeg.input(input_path)
        added_audio = ffmpeg.input(audio_path)

        merged_audio = ffmpeg.filter([input_video.audio, added_audio], "amix")

        (
            ffmpeg.concat(input_video, merged_audio, v=1, a=1)
            .output(output_path)
            .run(overwrite_output=True)
        )


class FFProbeTool(Tool):
    name = "ffprobe_tool"
    description = """
    This tool extracts metadata from input video using ffmpeg/ffprobe
    Input is input_path.
    Output is video metadata as a string.
    """
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, input_path: str):
        video_info = get_video_info(input_path)
        return json.dumps(video_info, indent=2) if video_info else None


class ImageToVideoTool(Tool):
    name = "image_to_video_tool"
    description = """
    This tool generates an N-second video clip from an image.
    Inputs are image_path, duration, output_path.
    """
    inputs = ["text", "text", "integer", "integer"]
    outputs = ["None"]

    def __call__(
        self, image_path: str, output_path: str, duration: int, framerate: int = 24
    ):
        (
            ffmpeg.input(image_path, loop=1, t=duration, framerate=framerate)
            .output(output_path, vcodec="libx264")
            .overwrite_output()
            .run()
        )


class ImageDirectoryToVideoTool(Tool):
    name = "image_directory_video_tool"
    description = """
    This tool creates video
    from a directory of images. Inputs
    are input_path and output_path. 
    Output is the output_path.
    """
    inputs = ["text", "text"]
    outputs = ["text"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        framerate: int = 24,
        extension: str = "jpg",
    ):
        # Check for valid extension
        valid_extensions = ["jpg", "png", "jpeg"]
        if extension not in valid_extensions:
            raise ValueError(
                f"Invalid extension {extension}. Must be one of {valid_extensions}"
            )

        (
            ffmpeg.input(
                input_path.rstrip("/") + "/*." + extension.lstrip("."),
                pattern_type="glob",
                framerate=framerate,
            )
            .output(output_path)
            .overwrite_output()
            .run()
        )
        return output_path


class ImageZoomPanTool(Tool):
    name = "image_zoompan_tool"
    description = """
    This tool creates a video by applying the zoompan filter for a Ken Burns effect on image.
    Inputs are input_path, output_path, zoom_factor.
    """
    inputs = ["text", "text", "integer"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, zoom_factor: int = 25):
        (
            ffmpeg.input(input_path)
            .output(output_path, vf="zoompan=z='zoom+0.001':d={}".format(zoom_factor))
            .run()
        )


class SpeechToSubtitleTool(Tool):
    name = "speech_to_subtitle_tool"
    description = """
    This tool generates .srt (SubRip) caption files using speech-to-text (STT). 
    Inputs are input_path for audio/video sources and output_path for caption file.
    highlight_words defaults to True, which highlights words in the caption file.
    """

    inputs = ["text", "text", "boolean"]
    outputs = ["None"]

    def __init__(
        self, language="en", device="cpu", model_name="small", compute_type="int8"
    ):
        self.device = device
        self.compute_type = compute_type
        self.model = whisperx.load_model(
            model_name, self.device, compute_type=self.compute_type
        )
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )

    def convert_to_srt_time(self, time: float) -> str:
        hh, rem = divmod(time, 3600)
        mm, ss = divmod(rem, 60)
        ms = str(time).split(".")[1]
        return "{:02d}:{:02d}:{:02d},{}".format(int(hh), int(mm), int(ss), ms)

    def create_srt_content(self, result: dict) -> str:
        srt_content = []
        for idx, js in enumerate(result["segments"]):
            start_time, end_time = (
                self.convert_to_srt_time(js["start"]),
                self.convert_to_srt_time(js["end"]),
            )
            srt_content.append(
                "{}\n{} --> {}\n{}\n\n".format(
                    idx + 1, start_time, end_time, js["text"]
                )
            )
        return "".join(srt_content)

    def write_to_srt(self, srt_path: str, srt_content: str):
        with open(srt_path, "w") as outfile:
            outfile.write(srt_content)

    def transcribe_audio(self, input_path: str, batch_size: int = 16):
        audio = whisperx.load_audio(input_path)
        result = self.model.transcribe(audio, batch_size=batch_size)
        return result

    def align_audio(self, segments, audio):
        return whisperx.align(
            segments,
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

    def generate_srt(self, input_path: str, srt_path: str, batch_size: int = 16):
        transcribed_result = self.transcribe_audio(input_path, batch_size)
        aligned_result = self.align_audio(
            transcribed_result["segments"], audio=whisperx.load_audio(input_path)
        )
        srt_content = self.create_srt_content(aligned_result)
        self.write_to_srt(srt_path, srt_content)

    def __call__(
        self,
        input_path,
        output_path,
        highlight_words: bool = True,
        batch_size: int = 16,
    ):
        if not highlight_words:
            self.generate_srt(input_path, output_path)
        else:
            audio = whisperx.load_audio(input_path)
            result = self.model.transcribe(audio, batch_size=batch_size)
            result = whisperx.align(
                result["segments"],
                self.model_a,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            with open(output_path, "w") as outfile:
                for idx, segment in enumerate(result["segments"]):
                    start_time_srt = self.convert_to_srt_time(segment["start"])
                    end_time_srt = self.convert_to_srt_time(segment["end"])

                    words = []
                    for word_timing in segment["words"]:
                        words.append(word_timing["word"])

                    for i, word in enumerate(words):
                        highlighted_text = " ".join(
                            ["<u>" + word + "</u>" if word == w else w for w in words]
                        )

                        outfile.write(
                            "{}\n{} --> {}\n{}\n\n".format(
                                idx + 1, start_time_srt, end_time_srt, highlighted_text
                            )
                        )


class VideoAutoCropTool(Tool):
    name = "auto_crop_tool"
    description = """
    This tool automatically crops a video.
    Inputs are input_path as a string and output_path as a string.
    Output is the output_path.
    """
    inputs = ["text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str):
        video_info = get_video_info(input_path)
        samples = [float(video_info["duration"]) * i / 4 for i in range(1, 4)]

        crop_params_list = []
        for sample in samples:
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                input_path,
                "-vf",
                "cropdetect=24:16:0",
                "-vframes",
                "1",
                "-ss",
                str(sample),
                "-f",
                "null",
                "-",
            ]

            pipe = subprocess.Popen(ffmpeg_command, stderr=subprocess.PIPE)
            out, err = pipe.communicate()
            crop_match = re.search(r"crop=(\d+:\d+:\d+:\d+)", err.decode("utf-8"))

            if not crop_match:
                continue

            crop_params = crop_match.group(1)
            crop_params_list.append(crop_params)

        if not crop_params_list:
            raise ValueError("Could not find crop parameters in any sample frames.")

        most_common_crop_params = Counter(crop_params_list).most_common(1)[0][0]
        w, h, x, y = map(int, most_common_crop_params.split(":"))
        ffmpeg.input(input_path).output(
            output_path, vf="crop={}:{}:{}:{}".format(w, h, x, y)
        ).run(overwrite_output=True)


class VideoCaptionTool(Tool):
    name = "video_caption_tool"
    description = """
    This tool subtitles/captions a video with a text overlay from a .srt subtitle file. 
    Inputs are input_path, output_path, srt_path.
    """
    inputs = ["text", "text", "text", "text"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        srt_path: str,
        subtitle_style: str = None,
    ):
        if input_path == output_path:
            output_path = modify_file_name(output_path, "cap_")
        subtitle_arg = f"subtitles={srt_path}"
        if subtitle_style:
            subtitle_arg += f":force_style='{subtitle_style}'"
        (
            ffmpeg.input(input_path)
            .output(output_path, vf=subtitle_arg)
            .overwrite_output()
            .run()
        )


class VideoCropTool(Tool):
    name = "video_crop_tool"
    description = """
    This tool crops a video with inputs: 
    input_path, output_path, 
    top_x, top_y, 
    bottom_x, bottom_y.
    Output is the output_path.
    """
    inputs = ["text", "text", "text", "text", "text", "text"]
    outputs = ["text"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        top_x: str,
        top_y: str,
        bottom_x: str,
        bottom_y: str,
    ):
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.crop(
            stream,
            int(top_y),
            int(top_x),
            int(bottom_y) - int(top_y),
            int(bottom_x) - int(top_x),
        )
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)
        return output_path


class VideoFlipTool(Tool):
    name = "video_flip_tool"
    description = """
    This tool flips video along the horizontal 
    or vertical axis. Inputs are input_path, 
    output_path and orientation. Output is output_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def __call__(
        self, input_path: str, output_path: str, orientation: str = "horizontal"
    ):
        # Check for valid orientation
        valid_orientations = ["horizontal", "vertical"]
        if orientation not in valid_orientations:
            raise ValueError(
                f"Invalid orientation {orientation}. Must be one of {valid_orientations}"
            )

        flip = ffmpeg.vflip if orientation == "vertical" else ffmpeg.hflip
        stream = ffmpeg.input(input_path)
        stream = flip(stream)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)
        return output_path


class VideoFrameClassifierTool(Tool):
    name = "video_frame_classifier_tool"
    description = """
    This tool classifies frames from video input using a given ONNX model.
    Inputs are input_path, model_path, and n (infer every nth frame, skip n frames).
    """
    inputs = ["text", "text", "integer"]
    outputs = ["None"]

    def get_video_size(self, filename):
        video_info = get_video_info(filename)
        width = int(video_info["width"])
        height = int(video_info["height"])
        return width, height

    def get_metadata_from_csv(self, metadata_path):
        with open(metadata_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels = row["labels"].split("|")
                input_shape = eval(row["input_shape"])
                return labels, input_shape

    def classify_frame(self, frame):
        input_name = self.onnx_session.get_inputs()[0].name
        raw_result = self.onnx_session.run(None, {input_name: frame})
        result_indices = np.argmax(raw_result[0], axis=1)
        results = np.take(self.labels, result_indices, axis=0)
        return str(results[0])

    def __call__(self, input_path: str, model_path: str, n: int = 5):
        self.onnx_session = rt.InferenceSession(model_path)
        self.labels, self.input_shape = self.get_metadata_from_csv(
            os.path.join(os.path.dirname(model_path), "metadata.csv")
        )
        width, height = self.get_video_size(input_path)
        out, _ = (
            ffmpeg.input(input_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        results = {}
        for i in range(0, len(video), n):
            frame = video[i].astype("float32") / 255.0
            frame = frame * 2 - 1
            frame_resized = cv2.resize(
                frame, self.input_shape[1:3]
            )  # Adjust according to your model's input size
            frame_ready = np.expand_dims(frame_resized, axis=0)
            results[i] = self.classify_frame(frame_ready)

        # Save results to json file
        json_output_path = os.path.splitext(input_path)[0] + ".json"
        with open(json_output_path, "w") as json_file:
            json.dump(results, json_file)


class VideoFrameSampleTool(Tool):
    name = "video_frame_sample_tool"
    description = """
    This tool samples an image frame from an input video. 
    Inputs are input_path, output_path, and frame_number.
    Output is the output_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def __call__(self, input_path: str, output_path: str, frame_number: int):
        out, _ = (
            ffmpeg.input(input_path)
            .filter("select", "gte(n,{})".format(str(frame_number)))
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True)
        )
        img = Image.open(BytesIO(out))
        img.save(output_path)
        return output_path


class VideoGopChunkerTool(Tool):
    name = "video_chunker_tool"
    description = """
    This tool segments video input into GOPs (Group of Pictures) chunks of 
    segment_length (in seconds). Inputs are input_path and segment_length.
    """
    inputs = ["text", "integer"]
    outputs = ["None"]

    def __init__(self):
        super().__init__()

    def __call__(self, input_path, segment_length):
        basename = Path(input_path).stem
        output_dir = Path(input_path).parent
        video_info = get_video_info(input_path)
        num_segments = math.ceil(float(video_info["duration"]) / segment_length)
        num_digits = len(str(num_segments))
        filename_pattern = f"{output_dir}/{basename}_%0{num_digits}d.mp4"

        ffmpeg.input(input_path).output(
            filename_pattern,
            c="copy",
            map="0",
            f="segment",
            segment_time=segment_length,
        ).run()


class VideoHTTPServerTool(Tool):
    name = "video_http_server_tool"
    description = """
    This tool streams a source video to an HTTP server. 
    Inputs are input_path and server_url.
    """
    inputs = ["text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, server_url: str = "http://localhost:8080"):
        process = (
            ffmpeg.input(input_path)
            .output(
                server_url,
                codec="copy",  # use same codecs of the original video
                listen=1,  # enables HTTP server
                f="flv",
            )  # ffplay -f flv http://localhost:8080
            .global_args("-re")  # argument to act as a live stream
            .overwrite_output()
            .run()
        )


class VideoLetterBoxingTool(Tool):
    name = "video_letterboxing_tool"
    description = """
    This tool adds letterboxing to a video.
    Inputs are input_path, output_path, width, height, bg_color.
    """
    inputs = ["text", "text", "int", "int", "text"]
    outputs = ["None"]

    def __call__(
        self, input_path: str, output_path: str, width: int = 1920, height: int = 1080, bg_color: str = 'black'):
        video_info = get_video_info(input_path)
        old_width = int(video_info["width"])
        old_height = int(video_info["height"])

        # Check if the video is in portrait mode
        if old_height >= old_width:
            vf_option = "scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:-1:-1:color={}".format(
                width, height, width, height, bg_color
            )
        else:
            vf_option = "scale={}:-1".format(width)
        (ffmpeg.input(input_path).output(output_path, vf=vf_option).run())


class VideoOverlayTool(Tool):
    name = "video_overlay_tool"
    description = """
    This tool overlays one video on top of another.
    Inputs are main_video_path, overlay_video_path, output_path, x_position, y_position.
    """
    inputs = ["text", "text", "text", "integer", "integer"]
    outputs = ["None"]

    def __call__(
        self,
        main_video_path: str,
        overlay_video_path: str,
        output_path: str,
        x_position: int,
        y_position: int,
    ):
        main = ffmpeg.input(main_video_path)
        overlay = ffmpeg.input(overlay_video_path)

        (
            ffmpeg.output(
                ffmpeg.overlay(main, overlay, x=x_position, y=y_position), output_path
            )
            .overwrite_output()
            .run()
        )


class VideoReverseTool(Tool):
    name = "video_reverse_tool"
    description = """
    This tool reverses a video. 
    Inputs are input_path and output_path.
    """
    inputs = ["text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str):
        (
            ffmpeg.input(input_path)
            .filter_("reverse")
            .output(output_path)
            .overwrite_output()
            .run()
        )


class VideoResizeTool(Tool):
    name = "video_resize_tool"
    description = """
    This tool resizes the video to the specified dimensions.
    Inputs are input_path, width, height, output_path.
    """
    inputs = ["text", "text", "integer", "integer"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, width: int, height: int):
        (
            ffmpeg.input(input_path)
            .output(output_path, vf="scale={}:{}".format(width, height))
            .overwrite_output()
            .run()
        )


class VideoRotateTool(Tool):
    name = "video_rotate_tool"
    description = """
    This tool rotates a video by a specified angle. 
    Inputs are input_path, output_path and rotation_angle in degrees.
    """
    inputs = ["text", "text", "integer"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, rotation_angle: int):
        (
            ffmpeg.input(input_path)
            .filter_("rotate", rotation_angle)
            .output(output_path)
            .overwrite_output()
            .run()
        )


class VideoSceneSplitTool(Tool):
    name = "scene_split_tool"
    description = """
    This tool performs scene detection and splitting. 
    Inputs are input_path.
    """
    inputs = ["text"]
    outputs = ["None"]

    def __call__(self, input_path: str):
        scene_list = detect(input_path, ContentDetector())
        split_video_ffmpeg(input_path, scene_list)


class VideoSegmentDeleteTool(Tool):
    name = "segment_delete_tool"
    description = """
    This tool deletes a interval of video by timestamp.
    Inputs are input_path, output_path, start, end.
    Format start/end as float.
    """
    inputs = ["text", "text", "float", "float"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, start: float, end: float):
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                vf="select='not(between(t,{},{}))',setpts=N/FRAME_RATE/TB".format(
                    start, end
                ),
                af="aselect='not(between(t,{},{}))',asetpts=N/SR/TB".format(start, end),
            )
            .run()
        )


class VideoSpeedTool(Tool):
    name = "video_speed_tool"
    description = """
    This tool speeds up a video. 
    Inputs are input_path as a string, output_path as a string, speed_factor (float) as a string.
    Output is the output_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def __call__(self, input_path: str, output_path: str, speed_factor: float):
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.setpts(stream, "1/{}*PTS".format(float(speed_factor)))
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)
        return output_path


class VideoStabilizationTool(Tool):
    name = "video_stabilization_tool"
    description = """
    This tool stabilizes a video.
    Inputs are input_path, output_path, smoothing, zoom.
    """
    inputs = ["text", "text", "integer", "integer", "integer"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        smoothing: int = 10,
        zoom: int = 0,
        shakiness: int = 5,
    ):
        (
            ffmpeg.input(input_path)
            .output("null", vf="vidstabdetect=shakiness={}".format(shakiness), f="null")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                vf="vidstabtransform=smoothing={}:zoom={}:input={}".format(
                    smoothing, zoom, "transforms.trf"
                ),
            )
            .overwrite_output()
            .run()
        )


class VideoStackTool(Tool):
    name = "video_stack_tool"
    description = """
    This tool stacks two videos either vertically or horizontally based on the orientation parameter.
    Inputs are input_path, second_input, output_path, and orientation as strings.
    Output is the output_path.
    vertical orientation -> vstack, horizontal orientation -> hstack
    """
    inputs = ["text", "text", "text", "text"]
    outputs = ["None"]

    def __call__(
        self, input_path: str, second_input: str, output_path: str, orientation: str
    ):
        video1 = ffmpeg.input(input_path)
        video2 = ffmpeg.input(second_input)

        if orientation.lower() not in ["vstack", "hstack"]:
            raise ValueError("Orientation must be either 'vstack' or 'hstack'.")

        stacked = ffmpeg.filter((video1, video2), orientation)
        out = ffmpeg.output(stacked, output_path)
        out.run(overwrite_output=True)


class VideoTransitionTool(Tool):
    name = "xfade_transition_tool"
    description = """
    This tool applies a xfade filter transitions between two videos.
    Inputs are input_path1, input_path2, output_path, transition_type, duration, and offset.
    transition_type: fade, fadeblack, fadewhite, slideleft, slideright, slideup, slidedown,
    circlecrop, circleclose, circleopen, vertopen, horzopen, vertclose, horzclose, radial,
    coverleft, coverright, coverup, coverdown, squeezeev, squeezeeh, zoomin.
    """
    inputs = ["text", "text", "text", "text", "float", "float"]
    outputs = ["None"]

    def __call__(
        self,
        input_path1: str,
        input_path2: str,
        output_path: str,
        transition_type: str,
        duration: float,
        offset: float,
    ):
        (
            ffmpeg.input(input_path1)
            .output(
                output_path,
                filter_complex="xfade=transition={}:duration={}:offset={}".format(
                    transition_type, duration, offset
                ),
            )
            .global_args("-i", input_path2)
            .run()
        )


class VideoTrimTool(Tool):
    name = "video_trim_tool"
    description = """
    This tool trims a video. Inputs are input_path, output_path, 
    start_time, and end_time. Format start(end)_time: HH:MM:SS
    """
    inputs = ["text", "text", "text", "text"]
    outputs = ["None"]

    def __call__(
        self, input_path: str, output_path: str, start_time: str, end_time: str
    ):
        stream = ffmpeg.input(input_path)
        v = stream.trim(start=start_time, end=end_time).setpts("PTS-STARTPTS")
        if has_audio(input_path):
            a = stream.filter_("atrim", start=start_time, end=end_time).filter_(
                "asetpts", "PTS-STARTPTS"
            )
            joined = ffmpeg.concat(v, a, v=1, a=1).node
            out = ffmpeg.output(joined[0], joined[1], output_path)
        else:
            out = ffmpeg.output(v, output_path)
        out.run()


class VideoWatermarkTool(Tool):
    name = "video_watermark_tool"
    description = """
    This tool adds logo image as watermark to a video. 
    Inputs are input_path, output_path, watermark_path.
    """
    inputs = ["text", "text", "text", "integer", "integer"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        watermark_path: str,
        x: int = 10,
        y: int = 10,
    ):
        main = ffmpeg.input(input_path)
        logo = ffmpeg.input(watermark_path)
        (
            ffmpeg.filter([main, logo], "overlay", x, y)
            .output(output_path)
            .overwrite_output()
            .run()
        )
