#!/usr/bin/env python3
"""Observe operators through Python bytecode as an ecological substrate.

This module is a microscope slide, not a portable VM. Python bytecode is
CPython-version-specific (opcodes change between minor releases), so the
tools here are meant for local experiments — they assert structural
properties, not exact opcode strings.

Two complementary lenses on the same function:

* ``static_bytecode_tokens`` — every instruction the interpreter *could*
  execute. This is availability: the possible pathways the operator's body
  affords.

* ``traced_bytecode_tokens`` — the instructions actually executed during a
  call, captured via ``sys.settrace`` with ``frame.f_trace_opcodes``. This
  is activation: the pathway taken under a specific input.

The token grammar deliberately echoes the existing ``L:<name>`` stream:

    B:<OPNAME>@<offset>            instruction at byte offset
    CALL:<qualname>                resolved call target (best-effort)
    LINE:<lineno>                  source line transition (trace only)

Static and traced streams compose: their set-difference is the operator's
"latent" pathway — instructions present in the body but not exercised by
the current activation factor. That gap is the natural concentration-field
input: motifs that exist as potential but not as event.
"""

from __future__ import annotations

import dis
import sys
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class CodeBoundary:
    """The Python-given boundaries of a function's code object.

    These are the obvious cell walls the interpreter already maintains; we
    just surface them as fields so an experiment can treat a function as a
    bounded ecological compartment.
    """

    qualname: str
    filename: str
    firstlineno: int
    argcount: int
    nlocals: int
    names: tuple[str, ...]        # co_names: globals/attrs referenced
    varnames: tuple[str, ...]     # co_varnames: locals + args
    consts: tuple[Any, ...]       # co_consts: literal constants
    freevars: tuple[str, ...]     # co_freevars: closure cells

    @classmethod
    def of(cls, func: Callable[..., Any]) -> "CodeBoundary":
        code = _code_of(func)
        return cls(
            qualname=getattr(func, "__qualname__", code.co_name),
            filename=code.co_filename,
            firstlineno=code.co_firstlineno,
            argcount=code.co_argcount,
            nlocals=code.co_nlocals,
            names=tuple(code.co_names),
            varnames=tuple(code.co_varnames),
            consts=tuple(code.co_consts),
            freevars=tuple(code.co_freevars),
        )


def _code_of(func: Callable[..., Any]) -> CodeType:
    code = getattr(func, "__code__", None)
    if code is None or not isinstance(code, CodeType):
        raise TypeError(f"object has no __code__: {func!r}")
    return code


# ---------------------------------------------------------------------------
# Static lens — what the operator's body affords.
# ---------------------------------------------------------------------------
def static_bytecode_tokens(func: Callable[..., Any]) -> tuple[str, ...]:
    """Tokenize every instruction in ``func``'s code object.

    Each token is ``B:<OPNAME>@<offset>``. Offset is the byte offset of the
    instruction inside the code object — stable within one interpreter run
    but not guaranteed across CPython versions. Tests should assert
    structural properties (opname set, token shape, ordering by offset),
    not the exact sequence.
    """

    code = _code_of(func)
    return tuple(
        f"B:{ins.opname}@{ins.offset}" for ins in dis.get_instructions(code)
    )


def static_opnames(func: Callable[..., Any]) -> tuple[str, ...]:
    """The unique set of opnames in the function body, order-preserving."""

    seen: dict[str, None] = {}
    for ins in dis.get_instructions(_code_of(func)):
        seen.setdefault(ins.opname, None)
    return tuple(seen)


def static_call_targets(func: Callable[..., Any]) -> tuple[str, ...]:
    """Best-effort static call-target names from CALL-adjacent loads.

    Heuristic: between two CALL boundaries the *first* load that produces
    a name (LOAD_GLOBAL / LOAD_NAME / LOAD_ATTR / LOAD_METHOD / LOAD_DEREF)
    is the callable; later LOAD_FAST / LOAD_CONST are arguments. This
    surfaces motifs like ``FormulaGraphEdge`` or ``append`` without being
    fooled by argument loads. Call indirection still produces noise.
    """

    NAME_LIKE = (
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "LOAD_ATTR",
        "LOAD_METHOD",
        "LOAD_DEREF",
    )
    targets: list[str] = []
    pending: str | None = None
    for ins in dis.get_instructions(_code_of(func)):
        op = ins.opname
        if op in NAME_LIKE and ins.argval is not None and pending is None:
            pending = str(ins.argval)
        elif op in ("CALL", "CALL_FUNCTION_EX", "CALL_KW"):
            if pending is not None:
                targets.append(pending)
            pending = None
    return tuple(targets)


# ---------------------------------------------------------------------------
# Runtime lens — what the operator's body actually executes.
# ---------------------------------------------------------------------------
@dataclass
class TraceRecord:
    """A single observation from a traced call."""

    kind: str             # "line", "call", "opcode", "return"
    qualname: str
    lineno: int | None = None
    offset: int | None = None
    opname: str | None = None

    def as_token(self) -> str:
        if self.kind == "opcode":
            return f"B:{self.opname}@{self.offset}"
        if self.kind == "line":
            return f"LINE:{self.lineno}"
        if self.kind == "call":
            return f"CALL:{self.qualname}"
        if self.kind == "return":
            return f"RET:{self.qualname}"
        return f"{self.kind.upper()}:{self.qualname}"


@dataclass
class TraceResult:
    records: list[TraceRecord] = field(default_factory=list)
    return_value: Any = None

    def tokens(self) -> tuple[str, ...]:
        return tuple(r.as_token() for r in self.records)

    def opcode_tokens(self) -> tuple[str, ...]:
        return tuple(r.as_token() for r in self.records if r.kind == "opcode")

    def call_tokens(self) -> tuple[str, ...]:
        return tuple(r.as_token() for r in self.records if r.kind == "call")


def trace_call(
    func: Callable[..., Any],
    /,
    *args: Any,
    extra_codes: Sequence[CodeType] = (),
    opcodes: bool = True,
    **kwargs: Any,
) -> TraceResult:
    """Call ``func(*args, **kwargs)`` while recording bytecode-level events.

    Uses ``sys.monitoring`` (PEP 669, Python 3.12+). Instruction-level events
    are armed only on ``func``'s code object plus any ``extra_codes``; other
    frames contribute nothing (we deliberately stay local to keep traces
    interpretable). Set ``opcodes=False`` for line+call events only.
    """

    code = _code_of(func)
    qualname = getattr(code, "co_qualname", code.co_name)
    tracked: dict[int, tuple[CodeType, str]] = {
        id(code): (code, qualname),
    }
    for extra in extra_codes:
        eq = getattr(extra, "co_qualname", extra.co_name)
        tracked[id(extra)] = (extra, eq)

    result = TraceResult()
    offset_to_op = _offset_index(code)

    mon = sys.monitoring
    tool_id = mon.PROFILER_ID
    # If something else already owns this tool slot, fall back to DEBUGGER_ID.
    for candidate in (mon.PROFILER_ID, mon.DEBUGGER_ID, mon.OPTIMIZER_ID):
        try:
            mon.use_tool_id(candidate, "bytecode_genes")
            tool_id = candidate
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("no sys.monitoring tool id available")

    def on_instruction(c: CodeType, offset: int):
        entry = tracked.get(id(c))
        if entry is None:
            return mon.DISABLE
        opname = offset_to_op.get(offset) if c is code else _opname_at(c, offset)
        result.records.append(
            TraceRecord(
                kind="opcode",
                qualname=entry[1],
                offset=offset,
                opname=opname,
            )
        )

    def on_line(c: CodeType, lineno: int):
        entry = tracked.get(id(c))
        if entry is None:
            return mon.DISABLE
        result.records.append(
            TraceRecord(kind="line", qualname=entry[1], lineno=lineno)
        )

    def on_call(c: CodeType, ip: int, callable_obj: Any, arg0: Any):
        entry = tracked.get(id(c))
        if entry is None:
            return mon.DISABLE
        name = getattr(callable_obj, "__qualname__", None) or getattr(
            callable_obj, "__name__", repr(callable_obj)
        )
        result.records.append(TraceRecord(kind="call", qualname=name))

    def on_return(c: CodeType, ip: int, retval: Any):
        entry = tracked.get(id(c))
        if entry is None:
            return mon.DISABLE
        result.records.append(
            TraceRecord(kind="return", qualname=entry[1])
        )

    E = mon.events
    event_mask = E.LINE | E.CALL | E.PY_RETURN
    if opcodes:
        event_mask |= E.INSTRUCTION

    mon.register_callback(tool_id, E.INSTRUCTION, on_instruction)
    mon.register_callback(tool_id, E.LINE, on_line)
    mon.register_callback(tool_id, E.CALL, on_call)
    mon.register_callback(tool_id, E.PY_RETURN, on_return)
    try:
        for c, _qn in tracked.values():
            mon.set_local_events(tool_id, c, event_mask)
        try:
            result.return_value = func(*args, **kwargs)
        finally:
            for c, _qn in tracked.values():
                mon.set_local_events(tool_id, c, 0)
    finally:
        mon.register_callback(tool_id, E.INSTRUCTION, None)
        mon.register_callback(tool_id, E.LINE, None)
        mon.register_callback(tool_id, E.CALL, None)
        mon.register_callback(tool_id, E.PY_RETURN, None)
        mon.free_tool_id(tool_id)
    return result


def _offset_index(code: CodeType) -> dict[int, str]:
    return {ins.offset: ins.opname for ins in dis.get_instructions(code)}


def _opname_at(code: CodeType, offset: int) -> str | None:
    return _offset_index(code).get(offset)


# ---------------------------------------------------------------------------
# Static-vs-activated comparison: the latent pathway gap.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathwayDiff:
    """Sets of static-only and shared bytecode tokens for one function."""

    qualname: str
    static_total: int
    activated_total: int
    shared: tuple[str, ...]
    static_only: tuple[str, ...]  # available but not executed

    @property
    def activation_ratio(self) -> float:
        if not self.static_total:
            return 0.0
        return len(self.shared) / self.static_total


def pathway_diff(func: Callable[..., Any], trace: TraceResult) -> PathwayDiff:
    """Compare the function's static instruction stream to one trace."""

    qn = getattr(func, "__qualname__", _code_of(func).co_name)
    static = static_bytecode_tokens(func)
    # Keep only opcode tokens from the matching qualname in the trace.
    activated = tuple(
        r.as_token()
        for r in trace.records
        if r.kind == "opcode" and r.qualname == qn
    )
    static_set = set(static)
    activated_set = set(activated)
    shared = tuple(t for t in static if t in activated_set)
    static_only = tuple(t for t in static if t not in activated_set)
    return PathwayDiff(
        qualname=qn,
        static_total=len(static_set),
        activated_total=len(activated_set),
        shared=shared,
        static_only=static_only,
    )


__all__ = [
    "CodeBoundary",
    "PathwayDiff",
    "TraceRecord",
    "TraceResult",
    "pathway_diff",
    "static_bytecode_tokens",
    "static_call_targets",
    "static_opnames",
    "trace_call",
]
