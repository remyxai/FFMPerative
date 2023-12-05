import shlex
import subprocess
import pkg_resources
from sys import argv

from . import tools as t

from interpretor import evaluate

tools = {
    "AudioAdjustmentTool": t.AudioAdjustmentTool(),
    "AudioVideoMuxTool": t.AudioVideoMuxTool(),
    "FFProbeTool": t.FFProbeTool(),
    "ImageDirectoryToVideoTool": t.ImageDirectoryToVideoTool(),
    "ImageToVideoTool": t.ImageToVideoTool(),
    "VideoCropTool": t.VideoCropTool(),
    "VideoFlipTool": t.VideoFlipTool(),
    "VideoFrameSampleTool": t.VideoFrameSampleTool(),
    "VideoGopChunkerTool": t.VideoGopChunkerTool(),
    "VideoHTTPServerTool": t.VideoHTTPServerTool(),
    "VideoLetterBoxingTool": t.VideoLetterBoxingTool(),
    "VideoOverlayTool": t.VideoOverlayTool(),
    "VideoResizeTool": t.VideoResizeTool(),
    "VideoReverseTool": t.VideoReverseTool(),
    "VideoRotateTool": t.VideoRotateTool(),
    "VideoSegmentDeleteTool": t.VideoSegmentDeleteTool(),
    "VideoSpeedTool": t.VideoSpeedTool(),
    "VideoStackTool": t.VideoStackTool(),
    "VideoTrimTool": t.VideoTrimTool(),
    "VideoWatermarkTool": t.VideoWatermarkTool(),
}


def run(prompt):
    ffmp_path = pkg_resources.resource_filename("ffmperative", "bin/ffmp")
    safe_prompt = shlex.quote(prompt)
    command = [ffmp_path, "-p", safe_prompt]

    try:
        # Run the command without using shell
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Get the standard output and split into lines
        output = result.stdout
        return output.split("### Assistant:")[2]
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., log them, raise an exception, or return a default value)
        print(f"Error occurred: {e}")
        return None


def ffmp(prompt, tools=tools):
    parsed_output = run(prompt)
    parsed_ast = ast.parse(parsed_output)
    result = evaluate(parsed_ast, tools)
    return result
