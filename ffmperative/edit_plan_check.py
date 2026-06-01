"""Pre-execution edit-plan verification for FFMPerative.

Adapted from *Aurora: Unified Video Editing with a Tool-Using Agent*
(arXiv:2605.18748). Aurora's VLM agent maps a raw user request into a
**structured edit plan aligned with the tool's conditioning channels**,
resolving textual and visual underspecification *before* generation. Aurora
shows that catching incomplete plans up front — rather than feeding a
half-specified request to the generator — measurably improves
instruction-following.

This module brings that idea to FFMPerative. The LLM agent emits a sequence
of tool calls; before the interpretor executes them, we parse that sequence
into a structured plan and surface underspecification — unknown tools and
missing required arguments — so the gaps can be flagged (or, downstream,
repaired) instead of failing silently mid-pipeline.

The generative / diffusion-transformer half of Aurora is intentionally out
of scope: FFMPerative composes ``ffmpeg`` primitives, so the value is the
plan-completeness check over those primitives, not video synthesis.
"""

import ast
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union


@dataclass
class PlanStep:
    """A single tool call in the agent's edit plan, with its completeness."""

    tool: str
    args_provided: List[str]
    missing_required: List[str]
    is_known_tool: bool

    @property
    def is_underspecified(self) -> bool:
        return (not self.is_known_tool) or bool(self.missing_required)


@dataclass
class EditPlanReport:
    """Structured view of an agent edit plan and where it is underspecified."""

    steps: List[PlanStep] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when every step names a known tool with all required args."""
        return all(not step.is_underspecified for step in self.steps)

    @property
    def issues(self) -> List[str]:
        """Human-readable description of each underspecified step."""
        messages = []
        for step in self.steps:
            if not step.is_known_tool:
                messages.append("unknown tool `{}`".format(step.tool))
            elif step.missing_required:
                messages.append(
                    "`{}` is missing required argument(s): {}".format(
                        step.tool, ", ".join(step.missing_required)
                    )
                )
        return messages


def _required_params(tool: Callable) -> List[str]:
    """Names of parameters a tool *must* receive (no default value).

    Tools are ``Tool`` instances, so we inspect the bound ``__call__``; plain
    callables are inspected directly. ``*args`` / ``**kwargs`` and ``self`` are
    ignored.
    """
    target = getattr(tool, "__call__", tool)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return []

    required = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


def check_edit_plan(
    code: Union[str, ast.AST], tools: Dict[str, Callable]
) -> EditPlanReport:
    """Parse the agent's tool sequence into a structured, verified edit plan.

    Mirrors Aurora's "complete edit planning" step: every tool call is checked
    for completeness against the available ``tools`` *before* execution.

    Args:
        code: The agent's generated tool sequence, either as source text or an
            already-parsed AST (the interpretor passes whichever it holds).
        tools: Mapping of tool name -> callable, as built by
            :func:`ffmperative.tool_mapping.generate_tools_mapping`.

    Returns:
        An :class:`EditPlanReport`; empty when nothing parseable is found.
    """
    report = EditPlanReport()

    if isinstance(code, ast.AST):
        tree = code
    elif isinstance(code, str):
        if not code.strip():
            return report
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return report
    else:
        return report

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue

        name = node.func.id
        known = name in tools
        provided_kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
        positional_count = len(node.args)

        missing: List[str] = []
        if known:
            required = _required_params(tools[name])
            # Positional args fill required params left-to-right; keyword args
            # fill by name. Whatever required name is left unfilled is missing.
            filled = set(required[:positional_count]) | set(provided_kwargs)
            missing = [param for param in required if param not in filled]

        report.steps.append(
            PlanStep(
                tool=name,
                args_provided=provided_kwargs,
                missing_required=missing,
                is_known_tool=known,
            )
        )

    return report
