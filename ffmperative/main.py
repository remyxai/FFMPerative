from interpretor import *
from sys import argv

def run_ffmp(prompt):
    # Run the ffmp binary with the prompt
    command = '/v_4TB_2/llamafile_experiments/llamafile/ffmp -p "{}"'.format(prompt)
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    # Get the standard output and split into lines
    output = result.stdout
    return output.split("### Assistant:")[2]

# Example usage
prompt = argv[1] #"I want to rotate 'summerVid.mov' by 23 degrees"
tools = {
    "AudioAdjustmentTool": AudioAdjustmentTool(),
    "AudioVideoMuxTool": AudioVideoMuxTool(),
    "FFProbeTool": FFProbeTool(),
    "ImageDirectoryToVideoTool": ImageDirectoryToVideoTool(),
    "ImageToVideoTool":  ImageToVideoTool(),
    "VideoCropTool": VideoCropTool(),
    "VideoFlipTool": VideoFlipTool(),
    "VideoFrameSampleTool":  VideoFrameSampleTool(),
    "VideoGopChunkerTool": VideoGopChunkerTool(),
    "VideoHTTPServerTool": VideoHTTPServerTool(),
    "VideoLetterBoxingTool": VideoLetterBoxingTool(),
    "VideoOverlayTool": VideoOverlayTool(),
    "VideoResizeTool": VideoResizeTool(),
    "VideoReverseTool": VideoReverseTool(),
    "VideoRotateTool": VideoRotateTool(),
    "VideoSegmentDeleteTool": VideoSegmentDeleteTool(),
    "VideoSpeedTool": VideoSpeedTool(),
    "VideoStackTool": VideoStackTool(),
    "VideoTrimTool": VideoTrimTool(),
    "VideoWatermarkTool": VideoWatermarkTool(),
}

parsed_output = run_ffmp(prompt)
parsed_ast = ast.parse(parsed_output)
result = evaluate(parsed_ast, tools)
print(result)
