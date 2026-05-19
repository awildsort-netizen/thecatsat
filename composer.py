#!/usr/bin/env python3
"""Minimal operator composition engine for SAT field workflows."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType
from typing import Callable, Iterable, Mapping, get_type_hints

FieldContext = dict[str, object]
Validator = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class MissingInput:
    key: str
    required_by: tuple[str, ...]


@dataclass(frozen=True)
class CompositionPlan:
    targets: tuple[str, ...]
    order: tuple[str, ...]
    missing: tuple[MissingInput, ...]


@dataclass(frozen=True)
class DependencyGraph:
    operators: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    missing: tuple[MissingInput, ...]


@dataclass(frozen=True)
class FieldOperator:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    run: Callable[[FieldContext], Mapping[str, object]]
    enabled: Callable[[FieldContext], bool] | None = None
    validate_inputs: Validator | None = None
    validate_outputs: Validator | None = None


@dataclass(frozen=True)
class OperatorCandidate:
    function: Callable[..., object]
    name: str
    module: str
    parameters: tuple[str, ...]
    return_type: object | None
    inferred_outputs: tuple[str, ...]
    locality_terms: tuple[str, ...]


@dataclass(frozen=True)
class ProviderFit:
    candidate: OperatorCandidate
    target: str
    score: float
    reasons: tuple[str, ...]


class Composer:
    """Plans and executes registered operators to satisfy requested targets."""

    def __init__(self, operators: Iterable[FieldOperator]):
        self._operators: dict[str, FieldOperator] = {}
        self._provider_by_output: dict[str, str] = {}
        for operator in operators:
            if operator.name in self._operators:
                raise ValueError(f"duplicate operator name: {operator.name}")
            self._operators[operator.name] = operator
            for output in operator.outputs:
                if output in self._provider_by_output:
                    existing = self._provider_by_output[output]
                    raise ValueError(
                        f"duplicate provider for output {output}: {existing}, {operator.name}"
                    )
                self._provider_by_output[output] = operator.name

    def plan(self, targets: Iterable[str], available_keys: Iterable[str] = ()) -> CompositionPlan:
        available = set(available_keys)
        target_list = tuple(dict.fromkeys(targets))
        missing_map: dict[str, set[str]] = {}
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def mark_missing(key: str, required_by: str) -> None:
            missing_map.setdefault(key, set()).add(required_by)

        def ensure_output(key: str, required_by: str) -> None:
            if key in available:
                return
            provider = self._provider_by_output.get(key)
            if provider is None:
                mark_missing(key, required_by)
                return
            ensure_operator(provider)

        def ensure_operator(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"operator dependency cycle detected at: {name}")
            visiting.add(name)
            operator = self._operators[name]
            for input_key in operator.inputs:
                ensure_output(input_key, name)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for target in target_list:
            ensure_output(target, "<target>")

        missing = tuple(
            MissingInput(key=key, required_by=tuple(sorted(required_by)))
            for key, required_by in sorted(missing_map.items())
        )
        return CompositionPlan(targets=target_list, order=tuple(ordered), missing=missing)

    def graph(self, targets: Iterable[str], available_keys: Iterable[str] = ()) -> DependencyGraph:
        plan = self.plan(targets, available_keys)
        operator_names = set(plan.order)
        edges: set[tuple[str, str]] = set()
        for name in plan.order:
            operator = self._operators[name]
            for input_key in operator.inputs:
                provider = self._provider_by_output.get(input_key)
                if provider is not None and provider in operator_names:
                    edges.add((provider, name))
        return DependencyGraph(
            operators=plan.order,
            edges=tuple(sorted(edges)),
            missing=plan.missing,
        )

    def run(self, targets: Iterable[str], initial_context: Mapping[str, object] | None = None) -> dict[str, object]:
        context: FieldContext = dict(initial_context or {})
        plan = self.plan(targets, context.keys())
        if plan.missing:
            details = ", ".join(
                f"{item.key} (required by {', '.join(item.required_by)})"
                for item in plan.missing
            )
            raise KeyError(f"missing inputs: {details}")

        for name in plan.order:
            operator = self._operators[name]
            if operator.enabled is not None and not operator.enabled(context):
                continue
            self._validate_operator_inputs(operator, context)
            output_values = operator.run(context)
            self._validate_operator_outputs(operator, output_values)
            for output_key in operator.outputs:
                context[output_key] = output_values[output_key]

        return {target: context[target] for target in dict.fromkeys(targets)}

    @staticmethod
    def _validate_operator_inputs(operator: FieldOperator, context: FieldContext) -> None:
        missing = [key for key in operator.inputs if key not in context]
        if missing:
            details = ", ".join(missing)
            raise KeyError(f"operator {operator.name} missing inputs: {details}")
        if operator.validate_inputs is not None:
            operator.validate_inputs(context)

    @staticmethod
    def _validate_operator_outputs(operator: FieldOperator, output_values: Mapping[str, object]) -> None:
        missing = [key for key in operator.outputs if key not in output_values]
        if missing:
            details = ", ".join(missing)
            raise KeyError(f"operator {operator.name} missing outputs: {details}")
        if operator.validate_outputs is not None:
            operator.validate_outputs(output_values)



def operator_candidate(function: Callable[..., object]) -> OperatorCandidate:
    signature = inspect.signature(function)
    type_hints = get_type_hints(function)
    parameters = tuple(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    )
    return_type = type_hints.get("return")
    name = function.__name__
    module = function.__module__
    return OperatorCandidate(
        function=function,
        name=name,
        module=module,
        parameters=parameters,
        return_type=return_type,
        inferred_outputs=infer_function_outputs(name, return_type),
        locality_terms=tokenize_terms(f"{module} {function.__qualname__}"),
    )


def discover_operator_candidates(module: ModuleType) -> tuple[OperatorCandidate, ...]:
    return tuple(
        operator_candidate(member)
        for _name, member in inspect.getmembers(module, inspect.isfunction)
        if member.__module__ == module.__name__
    )


def infer_function_outputs(name: str, return_type: object | None = None) -> tuple[str, ...]:
    if return_type is not None and is_dataclass(return_type):
        field_names = tuple(field.name for field in fields(return_type))
        if len(field_names) > 1:
            return field_names
    return (name,)


def rank_provider_candidates(
    target: str,
    candidates: Iterable[OperatorCandidate],
    nearby_terms: Iterable[str] = (),
    required_type: object | None = None,
) -> tuple[ProviderFit, ...]:
    fits = (
        provider_fit(target, candidate, nearby_terms, required_type)
        for candidate in candidates
    )
    return tuple(
        sorted(
            (fit for fit in fits if fit.score > 0.0),
            key=lambda fit: (-fit.score, fit.candidate.module, fit.candidate.name),
        )
    )


def choose_provider(
    target: str,
    candidates: Iterable[OperatorCandidate],
    nearby_terms: Iterable[str] = (),
    required_type: object | None = None,
) -> ProviderFit | None:
    return next(
        iter(rank_provider_candidates(target, candidates, nearby_terms, required_type)),
        None,
    )


def provider_fit(
    target: str,
    candidate: OperatorCandidate,
    nearby_terms: Iterable[str] = (),
    required_type: object | None = None,
) -> ProviderFit:
    target_terms = set(tokenize_terms(target))
    output_terms = set(term for output in candidate.inferred_outputs for term in tokenize_terms(output))
    return_terms = set(tokenize_type(candidate.return_type))
    parameter_terms = set(term for parameter in candidate.parameters for term in tokenize_terms(parameter))
    local_terms = set(candidate.locality_terms)
    nearby = set(term for term in nearby_terms if term)

    reason_scores = (
        (target in candidate.inferred_outputs, 4.0, "exact_output"),
        (target_terms <= output_terms and bool(target_terms), 2.5, "output_terms"),
        (bool(target_terms & return_terms), 1.5, "return_type"),
        (required_type is not None and candidate.return_type == required_type, 2.0, "required_type"),
        (bool(target_terms & local_terms), 0.75, "locality"),
        (bool(target_terms & parameter_terms), 0.5, "parameter_terms"),
        (bool(nearby & local_terms), 0.5, "nearby_locality"),
    )
    matched = tuple(reason for active, _score, reason in reason_scores if active)
    score = sum(score for active, score, _reason in reason_scores if active)
    return ProviderFit(candidate=candidate, target=target, score=score, reasons=matched)


def materialize_function_operator(
    candidate: OperatorCandidate,
    outputs: Iterable[str] | None = None,
    name: str | None = None,
) -> FieldOperator:
    output_keys = tuple(outputs or candidate.inferred_outputs)
    signature = inspect.signature(candidate.function)
    required_inputs = tuple(
        key
        for key in candidate.parameters
        if signature.parameters[key].default is inspect.Parameter.empty
    )

    def _run(context: FieldContext) -> Mapping[str, object]:
        kwargs = {
            key: context[key]
            for key in candidate.parameters
            if key in context or signature.parameters[key].default is inspect.Parameter.empty
        }
        result = candidate.function(**kwargs)
        if len(output_keys) == 1:
            return {output_keys[0]: result}
        if is_dataclass(result):
            return {key: getattr(result, key) for key in output_keys}
        if isinstance(result, Mapping):
            return {key: result[key] for key in output_keys}
        raise TypeError(f"operator {candidate.name} produced unsupported multi-output result")

    return FieldOperator(
        name=name or candidate.name,
        inputs=required_inputs,
        outputs=output_keys,
        validate_inputs=require_keys(required_inputs),
        run=_run,
    )


def tokenize_terms(value: object) -> tuple[str, ...]:
    words = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).split("_")
    return tuple(word.lower() for word in words if word)


def tokenize_type(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    type_name = getattr(value, "__name__", str(value))
    return tokenize_terms(type_name)


def require_keys(keys: Iterable[str]) -> Validator:
    expected = tuple(keys)

    def _validate(values: Mapping[str, object]) -> None:
        missing = [key for key in expected if key not in values]
        if missing:
            details = ", ".join(missing)
            raise KeyError(f"missing required keys: {details}")

    return _validate


