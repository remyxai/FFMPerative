"""Tests for Aurora-style edit-plan verification and its wiring into the
interpretor's execution gate.

These import the existing, non-new modules (`ffmperative.interpretor` and
`ffmperative.tool_mapping`) to prove the check is actually wired into the
plan-execution path, not just self-consistent in isolation.
"""

from ffmperative.interpretor import evaluate
from ffmperative.tool_mapping import generate_tools_mapping
from ffmperative.edit_plan_check import check_edit_plan, EditPlanReport


# --- check_edit_plan against the real FFMPerative tool registry ----------

def test_complete_plan_against_real_tools():
    tools = generate_tools_mapping()
    # VideoFlipTool requires input_path + output_path; orientation defaults.
    code = "VideoFlipTool(input_path='in.mp4', output_path='out.mp4')"
    report = check_edit_plan(code, tools)
    assert isinstance(report, EditPlanReport)
    assert report.is_complete
    assert report.issues == []


def test_missing_required_argument_is_flagged():
    tools = generate_tools_mapping()
    # AudioAdjustmentTool requires input_path, output_path, level -> drop level.
    code = "AudioAdjustmentTool(input_path='in.mp4', output_path='out.mp4')"
    report = check_edit_plan(code, tools)
    assert not report.is_complete
    assert any("level" in issue for issue in report.issues)


def test_positional_args_count_as_filled():
    tools = generate_tools_mapping()
    code = "VideoTrimTool('in.mp4', 'out.mp4', '00:00:01', '00:00:05')"
    report = check_edit_plan(code, tools)
    assert report.is_complete


def test_unknown_tool_is_flagged():
    tools = generate_tools_mapping()
    code = "TotallyMadeUpTool(input_path='in.mp4')"
    report = check_edit_plan(code, tools)
    assert not report.is_complete
    assert any("unknown tool" in issue for issue in report.issues)


# --- wiring: the check runs inside interpretor.evaluate before execution --

class _FakeAdjustTool:
    """Mimics a ffmperative Tool: required input_path/output_path/level."""

    def __call__(self, input_path, output_path, level):
        return output_path


def test_evaluate_warns_before_executing_underspecified_plan(capsys):
    tools = {"AudioAdjustmentTool": _FakeAdjustTool()}
    code = "x = AudioAdjustmentTool(input_path='a.mp4', output_path='b.mp4')"
    # Execution will raise because `level` is missing; the point is that the
    # underspecification warning is emitted *first*, by the wiring hook.
    try:
        evaluate(code, tools)
    except TypeError:
        pass
    out = capsys.readouterr().out
    assert "underspecified" in out.lower()
    assert "level" in out


def test_evaluate_runs_complete_plan_without_warning(capsys):
    tools = {"AudioAdjustmentTool": _FakeAdjustTool()}
    code = "x = AudioAdjustmentTool(input_path='a.mp4', output_path='b.mp4', level='0.5')"
    result = evaluate(code, tools)
    out = capsys.readouterr().out
    assert result == "b.mp4"
    assert "underspecified" not in out.lower()
