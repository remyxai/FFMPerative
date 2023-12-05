import shlex
import subprocess
import pkg_resources
from sys import argv

from . import tools as t
from .prompts import MAIN_PROMPT

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
    complete_prompt = MAIN_PROMPT.replace("<<prompt>>", prompt)
    safe_prompt = shlex.quote(complete_prompt)

    command = ["/ffmp/ffmp", "-p", safe_prompt]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        output = result.stdout
        return output.split("### Assistant:")[2]
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None

def ffmp(prompt, tools=tools):
    parsed_output = run(prompt)
    parsed_ast = ast.parse(parsed_output)
    result = evaluate(parsed_ast, tools)
    return result
