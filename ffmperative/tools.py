import json
import ffmpeg
from io import BytesIO
from PIL import Image
from transformers import Tool

from .utils import modify_file_name

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
        probe = ffmpeg.probe(input_path)
        return json.dumps(next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        ), indent=2)


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
        framerate: int = 25,
        extension: str = "jpg"
    ):
        # Check for valid extension
        valid_extensions = ["jpg", "png", "jpeg"]
        if extension not in valid_extensions:
            raise ValueError(f"Invalid extension {extension}. Must be one of {valid_extensions}")

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
            raise ValueError(f"Invalid orientation {orientation}. Must be one of {valid_orientations}")

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
        stream = ffmpeg.crop(stream, int(top_y), int(top_x), int(bottom_y) - int(top_y), int(bottom_x) - int(top_x))
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)
        return output_path


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


class VideoCompressionTool(Tool):
    name = "video_compression_tool"
    description = """
    This tool compresses input video/gif to optimized video/gif. 
    Inputs are input_path, output_path.
    Output is output_path.
    """
    inputs = ["text", "text"]
    outputs = ["text"]

    def __call__(self, input_path: str, output_path: str):
        if input_path == output_path:
            # Slightly change path to avoid overwriting input file
            output_path = modify_file_name(output_path, "compressed_")
        (
            ffmpeg.input(input_path)
            .filter_("split", "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse")
            .output(output_path)
            .overwrite_output()
            .run()
        )
        return output_path


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
        a = stream.filter_("atrim", start=start_time, end=end_time).filter_(
            "asetpts", "PTS-STARTPTS"    
        )
        joined = ffmpeg.concat(v, a, v=1, a=1).node
        out = ffmpeg.output(          
            joined[0],
            joined[1],
            output_path,
        )
        out.run()


class VideoFadeInTool(Tool):
    name = "video_fade_in_tool"
    description = """
    This tool applies a fade-in effect to a video.
    Inputs are input_path, output_path, fade_duration.
    """
    inputs = ["text", "text", "integer"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, fade_duration: int):
        (
            ffmpeg.input(input_path)
            .filter("fade", type="in", duration=fade_duration)
            .output(output_path)
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


class VideoCaptionTool(Tool):
    name = "video_caption_tool"
    description = """
    This tool subtitles/captions a video with a text overlay from a .srt subtitle file. 
    Inputs are input_path, output_path, srt_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, srt_path: str):
        if input_path == output_path:
            # Slightly change path to avoid overwriting input file
            output_path = modify_file_name(output_path, "cap_")
        (
            ffmpeg.input(input_path)
            .output(output_path, vf="subtitles={}".format(srt_path))
            .overwrite_output()
            .run()
        )


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
