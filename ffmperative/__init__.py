from .tools import *

try:
    from .extras import *

    _extras = True
except ImportError:
    _extras = False

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
        AudioVideoMuxTool(),
        FFProbeTool(),
        ImageDirectoryToVideoTool(),
        ImageToVideoTool(),
        VideoCropTool(),
        VideoFlipTool(),
        VideoFrameSampleTool(),
        VideoGopChunkerTool(),
        VideoHTTPServerTool(),
        VideoLetterBoxingTool(),
        VideoOverlayTool(),
        VideoResizeTool(),
        VideoReverseTool(),
        VideoRotateTool(),
        VideoSegmentDeleteTool(),
        VideoSpeedTool(),
        VideoStackTool(),
        VideoTrimTool(),
        VideoWatermarkTool(),
    ]

    if _extras:
        tools += [
            AudioDemuxTool(),
            ImageZoomPanTool(),
            SpeechToSubtitleTool(),
            VideoAutoCropTool(),
            VideoCaptionTool(),
            VideoFrameClassifierTool(),
            VideoSceneSplitTool(),
            VideoStabilizationTool(),
            VideoTransitionTool(),
        ]

    ffmp = HfAgent(url_endpoint, additional_tools=tools)
    return ffmp.run(prompt, run_prompt_template=template)
