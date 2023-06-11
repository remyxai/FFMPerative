import os
import sys
from .tools import *
from transformers.tools import HfAgent


def ffmp(
    prompt, url_endpoint="https://api-inference.huggingface.co/models/bigcode/starcoder"
):
    template = """
    Human tasks Assistant with a video processing workflows. Assistant uses all tools to generate an execution plan.

    Tools: <<all_tools>>

    Task: <<prompt>>

    Answer:
    """

    tools = [
        AudioAdjustmentTool(),
        FFProbeTool(),
        ImageDirectoryToVideoTool(),
        VideoFlipTool(),
        VideoFrameSampleTool(),
        VideoCropTool(),
        VideoSpeedTool(),
        VideoCompressionTool(),
        VideoResizeTool(),
        VideoTrimTool(),
        VideoFadeInTool(),
        VideoReverseTool(),
        VideoRotateTool(),
        VideoCaptionTool(),
        VideoWatermarkTool(),
        VideoHTTPServerTool(),
    ]

    ffmp = HfAgent(
        url_endpoint,
        additional_tools=tools,
    )
    return ffmp.run(prompt, run_prompt_template=template)
