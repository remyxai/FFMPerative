import re
import ast
import shlex
import subprocess
import pkg_resources
from sys import argv

from . import tools as t

from .interpretor import evaluate, extract_function_calls

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

def run(prompt, tools):
    safe_prompt = shlex.quote(prompt)
    command = '/ffmp/ffmp -c 6000 -p "{}"'.format(safe_prompt)

    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        output = result.stdout

        # Extract valid function calls using regular expressions
        pattern = '|'.join(re.escape(tool) for tool in tools.keys())
        matches = re.findall(r'(' + pattern + r'\(.*?\))', output)
        valid_output = ' '.join(matches)

        return valid_output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None

def ffmp(prompt, tools=tools):
    parsed_output = run(prompt, tools)
    if parsed_output:
        try:
            extracted_output = extract_function_calls(parsed_output, tools)
            parsed_ast = ast.parse(extracted_output)
            result = evaluate(parsed_ast, tools)
            return result
        except SyntaxError as e:
            print(f"Syntax error in parsed output: {e}")
    else:
        return None
