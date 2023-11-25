import os
import ffmpeg
import base64
import json
import requests
import subprocess

from pathlib import Path


def extract_and_encode_frame(video_path):
    # Get the duration of the video
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])

    # Calculate the timestamp for a frame in the middle of the video
    mid_time = duration / 2

    # Extract the frame at mid_time
    out, _ = (
        ffmpeg
        .input(video_path, ss=mid_time)
        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Encode the frame in base64
    base64_image = base64.b64encode(out).decode('utf-8')

    return base64_image

def process_video_directory(directory_path):
    json_list = []
    for filename in os.listdir(directory_path):
        print("Processing: ", filename)
        if filename.endswith((".mp4", ".avi", ".mov")):  # Add other video formats if needed
            video_path = os.path.join(directory_path, filename)
            base64_image = extract_and_encode_frame(video_path)
            json_list.append({"name": video_path, "sample": base64_image})
    return json_list

def post_json_to_endpoint(json_data, url):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=json_data, headers=headers)
    return response

def call_director(video_directory):
    json_data = process_video_directory(video_directory)

    # Endpoint URL
    endpoint_url = 'https://engine.remyx.ai/api/v1.0/task/b_roll/compose'

    # Make the POST request
    response = post_json_to_endpoint(json_data, endpoint_url)
    response = response.json()
    compose_plan = response["compose_plan"]
    join_command = response["join_command"]
    return compose_plan, join_command

def process_clip(clip_path):
    basename = os.path.basename(clip_path)
    processed_clip_path = Path("processed_clips") / basename
    subprocess.run(["ffmpeg", "-i", str(clip_path), "-vf", "scale=1920:1080,setsar=1,setdar=16/9,fps=30", str(processed_clip_path)])
    return processed_clip_path

def process_and_concatenate_clips(videos_string, output_path="composed_video.mp4"):
    # Split the string into individual paths
    video_paths = videos_string.strip().split()

    # Ensure there are video paths provided
    if not video_paths:
        raise ValueError("Please provide a string with video file paths")

    # Directory to store processed clips
    processed_clips_dir = Path("processed_clips")
    processed_clips_dir.mkdir(exist_ok=True)

    # Process each clip
    processed_clips = []
    for clip_path in video_paths:
        clip_path = Path(clip_path)
        if clip_path.exists() and clip_path.is_file():
            processed_clip = process_clip(clip_path)
            processed_clips.append(processed_clip)
        else:
            print(f"Warning: File not found {clip_path}")

    # Create a file list
    with open("files.txt", "w") as file_list:
        for clip in processed_clips:
            file_list.write(f"file '{clip}'\n")

    # Concatenate all processed clips
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "files.txt", output_path])

    # Cleanup
    for clip in processed_clips:
        clip.unlink()
    processed_clips_dir.rmdir()
    Path("files.txt").unlink()

    # Additionally, delete the original clip files
    for original_clip_path in video_paths:
        original_clip = Path(original_clip_path)
        if original_clip.exists():
            original_clip.unlink()

    return f"All clips processed and concatenated into {output_path}"

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
