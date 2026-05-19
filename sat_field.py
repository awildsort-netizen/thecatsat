#!/usr/bin/env python3
"""First-class formula graph adapters for SAT furnace CNF formulas."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Literal = tuple[int, bool]
Clause = tuple[Literal, ...]
CNF = list[Clause]
GraphNode = tuple[str, int]
Adjacency = dict[GraphNode, set[GraphNode]]


@dataclass(frozen=True)
class FormulaGraphEdge:
    source_kind: str
    source_id: int
    target_kind: str
    target_id: int
    polarity: int


@dataclass(frozen=True)
class FormulaGraph:
    edges: tuple[FormulaGraphEdge, ...]

    def to_adjacency(self) -> Adjacency:
        adjacency: Adjacency = {}
        for edge in self.edges:
            source = (edge.source_kind, edge.source_id)
            target = (edge.target_kind, edge.target_id)
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        return adjacency

    def to_csv_rows(self) -> list[dict[str, int | str]]:
        return [
            {
                "source_kind": edge.source_kind,
                "source_id": edge.source_id,
                "target_kind": edge.target_kind,
                "target_id": edge.target_id,
                "polarity": edge.polarity,
            }
            for edge in self.edges
        ]


def formula_graph(formula: CNF) -> FormulaGraph:
    edges: list[FormulaGraphEdge] = []
    for clause_id, clause in enumerate(formula):
        for variable, is_negated in clause:
            edges.append(
                FormulaGraphEdge(
                    source_kind="clause",
                    source_id=clause_id,
                    target_kind="variable",
                    target_id=variable,
                    polarity=-1 if is_negated else 1,
                )
            )
    return FormulaGraph(edges=tuple(edges))


def formula_graph_to_adjacency(graph: FormulaGraph) -> Adjacency:
    return graph.to_adjacency()


def spatial_rows_to_samples(spatial_rows: Sequence[dict[str, float | int | str]]):
    import sprite_detector

    return [
        sprite_detector.SpatialSample(
            t=int(row["t"]),
            kind=str(row["kind"]),
            id=int(row["id"]),
            heat=float(row["heat"]),
            influence=float(row["influence"]),
            entropy=float(row["entropy"]),
            spin=float(row["spin"]),
            pressure=float(row["pressure"]),
        )
        for row in spatial_rows
    ]


def write_formula_graph(path: Path, graph: FormulaGraph) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_kind", "source_id", "target_kind", "target_id", "polarity"],
        )
        writer.writeheader()
        writer.writerows(graph.to_csv_rows())
