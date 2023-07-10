import math
import json
import ffmpeg
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import Tool

from .utils import get_video_info, has_audio


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

        if has_audio(input_path):
            merged_audio = ffmpeg.filter([input_video.audio, added_audio], "amix")
            output_video = ffmpeg.concat(input_video, merged_audio, v=1, a=1)
        else:
            output_video = ffmpeg.concat(input_video, added_audio, v=1, a=1)
        output_video.output(output_path).run(overwrite_output=True)


class FFProbeTool(Tool):
    name = "ffprobe_tool"
    description = """
    This tool extracts metadata from input video using ffmpeg/ffprobe
    Input is input_path and output is video metadata as JSON.
    """
    inputs = ["text"]
    outputs = ["None"]

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
    outputs = ["None"]

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
        self,
        input_path: str,
        output_path: str,
        width: int = 1920,
        height: int = 1080,
        bg_color: str = "black",
    ):
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
