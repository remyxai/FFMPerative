import re
import os
import csv
import cv2
import json
import ffmpeg
import whisperx
import subprocess
import numpy as np
import onnxruntime as rt
import demucs.separate
from typing import Optional
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from collections import Counter
from transformers import Tool
from .utils import modify_file_name, get_video_info


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
        ms = str(time).split(".")[1][:3].ljust(3, "0")
        return "{:02d}:{:02d}:{:02d},{}".format(int(hh), int(mm), int(ss), ms)

    def format_timestamp(
        self,
        seconds: float,
        always_include_hours: bool = True,
        decimal_marker: str = ",",
    ):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

    def iterate_result(self, result: dict, options: dict):
        raw_max_line_width: Optional[int] = options["max_line_width"]
        max_line_count: Optional[int] = options["max_line_count"]
        highlight_words: bool = options["highlight_words"]
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segments = max_line_count is None or raw_max_line_width is None

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: list[dict] = []
            times = []
            last = result["segments"][0]["start"]
            for segment in result["segments"]:
                for i, original_timing in enumerate(segment["words"]):
                    timing = original_timing.copy()
                    long_pause = not preserve_segments
                    if "start" in timing:
                        long_pause = long_pause and timing["start"] - last > 3.0
                    else:
                        long_pause = False
                    has_room = line_len + len(timing["word"]) <= max_line_width
                    seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                    if line_len > 0 and has_room and not long_pause and not seg_break:
                        # line continuation
                        line_len += len(timing["word"])
                    else:
                        # new line
                        timing["word"] = timing["word"].strip()
                        if (
                            len(subtitle) > 0
                            and max_line_count is not None
                            and (long_pause or line_count >= max_line_count)
                            or seg_break
                        ):
                            # subtitle break
                            yield subtitle, times
                            subtitle = []
                            times = []
                            line_count = 1
                        elif line_len > 0:
                            # line break
                            line_count += 1
                            timing["word"] = "\n" + timing["word"]
                        line_len = len(timing["word"].strip())
                    subtitle.append(timing)
                    times.append(
                        (segment["start"], segment["end"], segment.get("speaker"))
                    )
                    if "start" in timing:
                        last = timing["start"]
            if len(subtitle) > 0:
                yield subtitle, times

        if "words" in result["segments"][0]:
            for subtitle, _ in iterate_subtitles():
                sstart, ssend, speaker = _[0]
                subtitle_start = self.format_timestamp(sstart)
                subtitle_end = self.format_timestamp(ssend)
                subtitle_text = " ".join([word["word"] for word in subtitle])
                has_timing = any(["start" in word for word in subtitle])

                # add [$SPEAKER_ID]: to each subtitle if speaker is available
                prefix = ""
                if speaker is not None:
                    prefix = f"[{speaker}]: "

                if highlight_words and has_timing:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        if "start" in this_word:
                            start = self.format_timestamp(this_word["start"])
                            end = self.format_timestamp(this_word["end"])
                            if last != start:
                                yield last, start, subtitle_text

                            yield start, end, prefix + " ".join(
                                [
                                    re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                    if j == i
                                    else word
                                    for j, word in enumerate(all_words)
                                ]
                            )
                            last = end
                else:
                    yield subtitle_start, subtitle_end, prefix + subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                if "speaker" in segment:
                    segment_text = f"[{segment['speaker']}]: {segment_text}"
                yield segment_start, segment_end, segment_text

    def create_srt_content(self, result: dict, highlight_words: bool) -> str:
        srt_content = []
        options = {
            "max_line_width": None,
            "max_line_count": None,
            "highlight_words": highlight_words,
        }
        for idx, (start, end, text) in enumerate(self.iterate_result(result, options)):
            srt_content.append(
                "{}\n{} --> {}\n{}\n\n".format(idx + 1, start, end, text)
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

    def generate_srt(
        self,
        input_path: str,
        srt_path: str,
        highlight_words: bool = True,
        batch_size: int = 16,
    ):
        transcribed_result = self.transcribe_audio(input_path, batch_size)
        aligned_result = self.align_audio(
            transcribed_result["segments"], audio=whisperx.load_audio(input_path)
        )
        srt_content = self.create_srt_content(aligned_result, highlight_words)
        self.write_to_srt(srt_path, srt_content)

    def __call__(
        self,
        input_path,
        output_path,
        highlight_words: bool = True,
        batch_size: int = 16,
    ):
        self.generate_srt(input_path, output_path, highlight_words, batch_size)


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
    Inputs are input_path, output_path, srt_path and optional input subtitle_style defaults to "Fontsize=24,PrimaryColour=&H0000ff&"
    """
    inputs = ["text", "text", "text", "text"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        srt_path: str,
        subtitle_style: Optional[str] = "Fontsize=24,PrimaryColour=&H0000ff&",
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
            frame_resized = cv2.resize(frame, self.input_shape[1:3])
            frame_ready = np.expand_dims(frame_resized, axis=0)
            results[i] = self.classify_frame(frame_ready)

        json_output_path = os.path.splitext(input_path)[0] + ".json"
        with open(json_output_path, "w") as json_file:
            json.dump(results, json_file)


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
