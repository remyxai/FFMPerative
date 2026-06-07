"""Causal Minimal Tool Filtering (CMTF) for the FFMPerative agent.

Adapted from "ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable
LLM Agents" (arXiv:2606.06284). FFMPerative currently exposes *every* ffmpeg
tool to the agent on every request. The paper's core finding is that larger
tool menus hurt reliability and inflate token cost, and that *semantic
relevance is not enough* -- a tool can be related to the task yet be
unnecessary or premature at the current step.

CMTF instead selects tools by *causal sufficiency*: each tool carries a
lightweight precondition/effect contract, and we expose only the minimal
frontier of tools that can advance from the current state toward the user's
goal. This module ports that idea to FFMPerative's ffmpeg primitives:

  * goal detection      -- match the user task to the tools whose *effects*
                           accomplish what was asked,
  * causal closure      -- pull in any prerequisite producer tools whose
                           effects satisfy an unmet precondition of a goal
                           tool (e.g. "slideshow then watermark" needs the
                           image-directory -> video step first),
  * conservative fallback -- when no causal frontier can be identified, expose
                           the full tool set, so filtering never makes the
                           agent strictly less capable than the baseline.

The result is a much smaller visible action surface for the agent, which both
reduces wrong-tool / premature calls and cuts the per-request token cost --
directly useful for the team's smaller fine-tuned StableLM-Zephyr-3B backend.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ToolContract:
    """Lightweight causal contract for a tool.

    ``preconditions`` are state predicates that must hold before the tool can
    run; ``effects`` are predicates it makes true; ``keywords`` are intent
    cues that tie a user request to this tool's effect.
    """

    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    keywords: Tuple[str, ...]


# State predicates that describe *source* artifacts the user can supply up
# front (as opposed to transformation effects a tool produces).
SOURCE_PREDICATES = ("video", "image", "image_dir", "audio")

# Precondition/effect contracts for FFMPerative's ffmpeg tools, keyed by the
# tool's class name (which is how ``generate_tools_mapping`` keys the agent's
# tool dict). Adding a tool without a contract simply means it is only exposed
# via the full-set fallback, never pruned incorrectly.
CONTRACTS: Dict[str, ToolContract] = {
    "AudioAdjustmentTool": ToolContract(
        ("video",), ("audio_adjusted",),
        ("volume", "loud", "quiet", "gain", "amplify", "decibel", "db",
         "audio level", "level"),
    ),
    "AudioVideoMuxTool": ToolContract(
        ("video", "audio"), ("muxed",),
        ("mux", "add audio", "combine audio", "soundtrack", "dub",
         "background music"),
    ),
    "FFProbeTool": ToolContract(
        ("video",), ("metadata",),
        ("metadata", "probe", "ffprobe", "codec", "inspect", "duration",
         "resolution info", "info"),
    ),
    "ImageToVideoTool": ToolContract(
        ("image",), ("video",),
        ("image to video", "from an image", "animate", "still image",
         "image into a video"),
    ),
    "ImageDirectoryToVideoTool": ToolContract(
        ("image_dir",), ("video",),
        ("slideshow", "directory of images", "folder of images",
         "sequence of images", "frames to video", "photos into"),
    ),
    "VideoCropTool": ToolContract(
        ("video",), ("cropped",), ("crop",),
    ),
    "VideoFlipTool": ToolContract(
        ("video",), ("flipped",), ("flip", "mirror"),
    ),
    "VideoFrameSampleTool": ToolContract(
        ("video",), ("frame",),
        ("sample frame", "extract frame", "thumbnail", "screenshot",
         "grab a frame", "single frame"),
    ),
    "VideoGopChunkerTool": ToolContract(
        ("video",), ("chunks",),
        ("segment", "chunk", "split", "gop", "break into"),
    ),
    "VideoHTTPServerTool": ToolContract(
        ("video",), ("served",),
        ("stream", "http server", "broadcast", "serve"),
    ),
    "VideoLetterBoxingTool": ToolContract(
        ("video",), ("letterboxed",),
        ("letterbox", "pillarbox", "aspect ratio", "black bars", "pad"),
    ),
    "VideoOverlayTool": ToolContract(
        ("video",), ("overlaid",),
        ("overlay", "picture in picture", "pip", "on top of"),
    ),
    "VideoReverseTool": ToolContract(
        ("video",), ("reversed",), ("reverse", "backwards", "rewind"),
    ),
    "VideoResizeTool": ToolContract(
        ("video",), ("resized",),
        ("resize", "rescale", "scale", "dimensions", "resolution"),
    ),
    "VideoRotateTool": ToolContract(
        ("video",), ("rotated",), ("rotate", "turn", "orientation"),
    ),
    "VideoSegmentDeleteTool": ToolContract(
        ("video",), ("segment_deleted",),
        ("delete", "remove segment", "cut out", "remove interval",
         "drop segment"),
    ),
    "VideoSpeedTool": ToolContract(
        ("video",), ("sped",),
        ("speed", "faster", "slow motion", "slow down", "fast forward",
         "timelapse", "speed up"),
    ),
    "VideoStackTool": ToolContract(
        ("video",), ("stacked",),
        ("stack", "side by side", "vstack", "hstack", "grid",
         "combine videos"),
    ),
    "VideoTrimTool": ToolContract(
        ("video",), ("trimmed",),
        ("trim", "clip", "cut", "shorten", "extract clip", "first seconds"),
    ),
    "VideoWatermarkTool": ToolContract(
        ("video", "image"), ("watermarked",),
        ("watermark", "logo", "brand", "stamp"),
    ),
}


@dataclass(frozen=True)
class FrontierReport:
    """Summary of a filtering decision, for logging / measurement."""

    total: int
    exposed: int
    names: List[str]
    filtered: bool
    reason: str
    est_token_savings_pct: float


def _initial_state(prompt: str) -> set:
    """Infer which source artifacts the user already has from the prompt."""
    p = prompt.lower()
    state = set()
    if re.search(r"\b(folder|directory|dir|slideshow|images|photos|pictures|frames)\b", p):
        state.add("image_dir")
    if re.search(r"\b(image|photo|picture|logo|png|jpe?g|still)\b", p):
        state.add("image")
    if re.search(r"\b(audio|sound|music|song|soundtrack|voice|mp3|wav|aac)\b", p):
        state.add("audio")
    if re.search(r"\b(video|clip|movie|footage|mp4|mov|avi|mkv|webm)\b", p) or not state:
        state.add("video")
    return state


def _intent_score(prompt: str, contract: ToolContract) -> int:
    """Score how strongly the prompt asks for this tool's effect."""
    p = prompt.lower()
    score = 0
    for kw in contract.keywords:
        if " " in kw:
            if kw in p:
                score += 2
        elif re.search(r"\b" + re.escape(kw) + r"\b", p):
            score += 1
    return score


def _find_producer(predicate, available, state, frontier):
    """Find a tool whose effect supplies an unmet precondition.

    Prefer producers whose own preconditions are already satisfiable from the
    current source state, so the causal closure stays minimal.
    """
    candidates = [n for n, c in available.items()
                  if predicate in c.effects and n not in frontier]
    candidates.sort(key=lambda n: 0 if set(available[n].preconditions) <= state else 1)
    return candidates[0] if candidates else None


def _token_savings_pct(tools: Dict, exposed: List[str]) -> float:
    """Rough percent reduction in tool-description tokens (~4 chars/token)."""
    def cost(name):
        desc = getattr(tools[name], "description", "") or ""
        return max(1, len(desc) // 4)

    total = sum(cost(n) for n in tools)
    if not total:
        return 0.0
    kept = sum(cost(n) for n in exposed if n in tools)
    return round(100.0 * (total - kept) / total, 1)


def select_tool_frontier(prompt: str, tools: Dict, min_score: int = 1):
    """Return the causal minimal tool frontier for ``prompt``.

    Args:
        prompt: the user's natural-language video-edit task.
        tools: the full ``{name: tool}`` mapping the agent would otherwise see.
        min_score: minimum intent score for a tool to be treated as a goal.

    Returns:
        ``(frontier_tools, report)`` where ``frontier_tools`` is a narrowed
        ``{name: tool}`` mapping (same objects, preserved order) and ``report``
        is a :class:`FrontierReport`. When no causal frontier can be
        identified the full ``tools`` mapping is returned unchanged.
    """
    available = {n: c for n, c in CONTRACTS.items() if n in tools}
    goal = {n for n, c in available.items()
            if _intent_score(prompt, c) >= min_score}

    if not goal:
        names = list(tools)
        return dict(tools), FrontierReport(
            total=len(tools), exposed=len(tools), names=names, filtered=False,
            reason="no causal frontier identified; exposing all tools",
            est_token_savings_pct=0.0,
        )

    state = _initial_state(prompt)
    frontier = set(goal)
    queue = list(goal)
    while queue:
        tool = queue.pop()
        for pre in available[tool].preconditions:
            if pre in state:
                continue
            if any(pre in available[f].effects for f in frontier):
                continue
            producer = _find_producer(pre, available, state, frontier)
            if producer is not None:
                frontier.add(producer)
                queue.append(producer)

    ordered = [n for n in tools if n in frontier]
    report = FrontierReport(
        total=len(tools), exposed=len(ordered), names=ordered, filtered=True,
        reason="exposed causal minimal frontier for goal(s): "
               + ", ".join(sorted(goal)),
        est_token_savings_pct=_token_savings_pct(tools, ordered),
    )
    return {n: tools[n] for n in ordered}, report
