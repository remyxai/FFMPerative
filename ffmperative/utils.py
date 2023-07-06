import os
import ffmpeg

from pathlib import Path


def modify_file_name(file_path, prefix):
    # Convert the file path to a Path object
    file_path = Path(file_path)

    # Extract the directory and the file name
    parent_dir = file_path.parent
    file_name = file_path.name

    # Add the prefix to the file name
    new_file_name = prefix + file_name

    # Create the new file path
    new_file_path = os.path.join(parent_dir, new_file_name)

    return new_file_path


def probe_video(input_path):
    return ffmpeg.probe(input_path)


def get_video_info(input_path):
    probe = probe_video(input_path)
    return next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )


def has_audio(input_path):
    probe = probe_video(input_path)
    return any(stream["codec_type"] == "audio" for stream in probe["streams"])
