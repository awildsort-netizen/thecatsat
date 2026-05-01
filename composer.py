#!/usr/bin/env python3
"""Minimal operator composition engine for SAT field workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

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


def require_keys(keys: Iterable[str]) -> Validator:
    expected = tuple(keys)

    def _validate(values: Mapping[str, object]) -> None:
        missing = [key for key in expected if key not in values]
        if missing:
            details = ", ".join(missing)
            raise KeyError(f"missing required keys: {details}")

    return _validate


def require_types(type_by_key: Mapping[str, type[Any] | tuple[type[Any], ...]]) -> Validator:
    requirements = dict(type_by_key)

    def _validate(values: Mapping[str, object]) -> None:
        for key, expected in requirements.items():
            if key not in values:
                continue
            if not isinstance(values[key], expected):
                expected_name = _type_name(expected)
                actual_name = type(values[key]).__name__
                raise TypeError(
                    f"key {key} expected type {expected_name}, got {actual_name}"
                )

    return _validate


def compose_validators(*validators: Validator | None) -> Validator:
    valid = tuple(validator for validator in validators if validator is not None)

    def _validate(values: Mapping[str, object]) -> None:
        for validator in valid:
            validator(values)

    return _validate


def _type_name(expected: type[Any] | tuple[type[Any], ...]) -> str:
    if isinstance(expected, tuple):
        return " | ".join(sorted(item.__name__ for item in expected))
    return expected.__name__
