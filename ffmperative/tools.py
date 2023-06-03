import ffmpeg
from io import BytesIO
from transformers import Tool


class AudioAdjustmentTool(Tool):
    name = "audio_adjustment_tool"
    description = """
    This tool modifies audio levels for an input video.
    Inputs are input_path, output_path, level (e.g. 0.5 or -13dB).
    """
    inputs = ["text", "text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, level: str):
        (
            ffmpeg.input(input_path)
            .output(output_path, af="volume={}".format(level))
            .run()
        )


class FFProbeTool(Tool):
    name = "ffprobe_tool"
    description = """
    This tool extracts metadata from input video using ffmpeg/ffprobe
    """
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, input_path: str):
        probe = ffmpeg.probe(input_path)
        return next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )


class ImageDirectory2VideoTool(Tool):
    name = "image_directory_video_tool"
    description = """
    This tool creates video
    from a directory of images. Inputs
    are input_path and output_path.
    """
    inputs = ["text", "text"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        framerate: int = 25,
        extension: str = "jpg",
    ):
        (
            ffmpeg.input(
                input_path + "/*" + extension, pattern_type="glob", framerate=framerate
            )
            .output(output_path)
            .run()
        )


class VideoFlipTool(Tool):
    name = "video_flip_tool"
    description = """
    This tool flips video along the horizontal 
    or vertical axis. Inputs are input_path, 
    output_path and orientation.
    """
    inputs = ["text", "text", "text"]
    outputs = ["None"]

    def __call__(
        self, input_path: str, output_path: str, orientation: str = "vertical"
    ):
        flip = ffmpeg.vflip if orientation == "vertical" else ffmpeg.hflip
        stream = ffmpeg.input(input_path)
        stream = flip(stream)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)


class VideoFrameSampleTool(Tool):
    name = "video_frame_sample_tool"
    description = """
    This tool samples an image frame from an input video. 
    Inputs are input_path, frame_number, and output_path.
    """
    inputs = ["text", "text", "integer"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, frame_number: int):
        out, _ = (
            ffmpeg.input(input_path)
            .filter("select", "gte(n,{})".format(frame_number))
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True)
        )
        img = Image.open(BytesIO(out))
        img.save(output_path)


class VideoCropTool(Tool):
    name = "video_crop_tool"
    description = """
    This tool crops a video with inputs: input_path, output_path, top_x, top_y, bottom_x, bottom_y.
    """
    inputs = ["text", "text", "integer", "integer", "integer", "integer"]
    outputs = ["None"]

    def __call__(
        self,
        input_path: str,
        output_path: str,
        top_x: int,
        top_y: int,
        bottom_x: int,
        bottom_y: int,
    ):
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.crop(stream, top_y, top_x, bottom_y - top_y, bottom_x - top_x)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)


class VideoSpeedTool(Tool):
    name = "video_speed_tool"
    description = """
    This tool speeds up a video. 
    Inputs are input_path, output_path, speed_factor.
    """
    inputs = ["text", "text", "float"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, speed_factor: float):
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.setpts(stream, "1/{}*PTS".format(speed_factor))
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)


class VideoCompressionTool(Tool):
    name = "video_compression_tool"
    description = """
    This tool compresses input video/gif to optimized video/gif. 
    Inputs are input_path, output_path.
    """
    inputs = ["text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str):
        (
            ffmpeg.input(input_path)
            .filter_("split", "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse")
            .output(output_path)
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
            .run()
        )


class VideoTrimTool(Tool):
    name = "video_trim_tool"
    description = """
    This tool trims a video. Inputs are input_path, output_path, start_time, and end_time.
    start(end)_time: HH:MM:SS
    """
    inputs = ["text", "text", "text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, start_time: str, end_time: str):
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.trim(stream, start=start_time, end=end_time)
        stream = ffmpeg.output(stream, output_path)
        ffmpeg.run(stream)



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
            .run()
        )


class VideoSubtitleTool(Tool):
    name = "video_subtitle_tool"
    description = """
    This tool adds a text overlay to a video from a .srt subtitle file. 
    Inputs are input_path, output_path, srt_path.
    """
    inputs = ["text", "text", "text"]
    outputs = ["None"]

    def __call__(self, input_path: str, output_path: str, srt_path: int):
        (
            ffmpeg.input(input_path)
            .output(output_path, vf="subtitles={}".format(srt_path))
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

    def __call__(self, input_path: str, output_path: str, watermark_path: int, x: int = 10, y: int = 10):
        main = ffmpeg.input(input_path)
        logo = ffmpeg.input(watermark_path)
        (
            ffmpeg
            .filter([main, logo], 'overlay', x, y)
            .output(output_path)
            .run()
        )
