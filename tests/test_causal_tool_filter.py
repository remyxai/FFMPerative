"""Integration tests for Causal Minimal Tool Filtering.

These exercise the same composition the ``ffmp()`` call site performs:
``select_tool_frontier`` narrows the tool mapping, and the existing
``ffmperative.interpretor`` (a non-new call-site module) only recognises the
tools left in that narrowed mapping. A heavyweight ffmpeg/PIL install is not
required for these; the real ``ffmp`` wiring is exercised separately when those
deps are available.
"""

import pytest

from ffmperative.interpretor import extract_function_calls
from ffmperative.causal_tool_filter import CONTRACTS, select_tool_frontier


class _StubTool:
    """Stand-in for a real ffmperative Tool, carrying only a description."""

    def __init__(self, name):
        self.description = f"This is the {name} tool. " * 4


def _full_tools():
    """A full {name: tool} mapping keyed exactly like the agent's real one."""
    return {name: _StubTool(name) for name in CONTRACTS}


def test_frontier_narrows_executable_surface():
    tools = _full_tools()
    frontier, report = select_tool_frontier("Please trim the first 10 seconds "
                                            "of my video", tools)

    # Causal frontier is a small subset that still contains the right tool.
    assert report.filtered is True
    assert "VideoTrimTool" in frontier
    assert report.exposed < report.total
    assert report.exposed <= 3
    assert report.est_token_savings_pct > 50.0

    # The narrowed mapping is what the existing interpretor would parse against:
    # an unrelated tool the model might hallucinate is no longer recognised.
    code = "x = VideoTrimTool(input_path='a.mp4', output_path='b.mp4')\n" \
           "y = VideoOverlayTool(main_video_path='b.mp4')"
    recognised = extract_function_calls(code, frontier)
    assert "VideoTrimTool(" in recognised
    assert "VideoOverlayTool(" not in recognised  # pruned from the frontier


def test_causal_closure_pulls_in_producer():
    # "slideshow ... watermark" needs the image-dir -> video producer step
    # before the watermark tool's precondition (a video) is satisfiable.
    tools = _full_tools()
    frontier, report = select_tool_frontier(
        "Build a slideshow from my photos and add my logo as a watermark", tools)

    assert "VideoWatermarkTool" in frontier
    assert "ImageDirectoryToVideoTool" in frontier  # causal prerequisite
    assert report.exposed < report.total


def test_fallback_exposes_all_when_no_frontier():
    tools = _full_tools()
    frontier, report = select_tool_frontier("hello there, how are you?", tools)

    assert report.filtered is False
    assert len(frontier) == len(tools)
    assert report.est_token_savings_pct == 0.0


def test_ffmp_uses_frontier_when_model_stays_in_bounds(monkeypatch):
    # Real call-site wiring: only runs where ffmpeg/PIL (tools.py deps) exist.
    pytest.importorskip("ffmpeg")
    pytest.importorskip("PIL")
    import ffmperative

    captured = {}

    # Spy on the interpretor's evaluate to capture exactly which tool mapping
    # ffmp() hands the agent's action layer -- without executing real ffmpeg.
    def fake_evaluate(parsed_ast, active_tools, *a, **k):
        captured["active_tools"] = active_tools
        return "ok"

    monkeypatch.setattr(ffmperative, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        ffmperative, "run_local",
        lambda prompt: "out = FFProbeTool(input_path='clip.mp4')",
    )

    result = ffmperative.ffmp("probe the metadata of clip.mp4")
    assert result == "ok"
    active = captured["active_tools"]
    # ffmp narrowed the executable surface to the causal frontier.
    assert "FFProbeTool" in active
    assert len(active) < len(ffmperative.tools)
