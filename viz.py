from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from manim import (
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    WHITE,
    YELLOW,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    Graph,
    Line,
    MathTex,
    NumberLine,
    Scene,
    Tex,
    VGroup,
    Write,
    DOWN,
    UP,
)


@dataclass
class TemporalSnapshot:
    """Represents a graph snapshot at a specific timestep."""

    time_index: int
    edges: List[Tuple[int, int]]


def generate_temporal_graph(
    num_nodes: int = 10,
    num_edges: int = 100,
    num_snapshots: int = 10,
) -> Tuple[List[int], List[TemporalSnapshot]]:
    """Create temporal snapshots that together contain ``num_edges`` edges."""

    if num_edges < num_snapshots:
        raise ValueError("Number of edges must be >= number of snapshots")

    nodes = list(range(num_nodes))
    random.seed(7)

    snapshots: List[TemporalSnapshot] = [
        TemporalSnapshot(time_index=i, edges=[]) for i in range(num_snapshots)
    ]

    # Guarantee at least one edge per snapshot so every frame has activity.
    for i in range(num_snapshots):
        u, v = random.sample(nodes, 2)
        snapshots[i].edges.append((u, v))

    for _ in range(num_edges - num_snapshots):
        u, v = random.sample(nodes, 2)
        t = random.randrange(num_snapshots)
        snapshots[t].edges.append((u, v))

    return nodes, snapshots


class TemporalGraphScene(Scene):
    """Visualize a temporal graph with 10 snapshots."""

    def construct(self) -> None:
        nodes, snapshots = generate_temporal_graph()

        header = Tex("Temporal Graph Evolution").to_edge(UP)
        timeline = NumberLine(
            x_range=[0, len(snapshots) - 1, 1],
            length=8,
            include_numbers=True,
        )
        timeline.next_to(header, DOWN, buff=0.5)
        tracker = Circle(radius=0.08, color=YELLOW, fill_opacity=1).move_to(
            timeline.number_to_point(0)
        )

        graph = self._create_graph(nodes)
        graph.next_to(timeline, DOWN, buff=0.8)

        edge_lines, edge_group = self._build_edge_layer(graph, snapshots)
        edge_group.set_z_index(graph.z_index - 1)

        self.add(edge_group)
        self.play(Write(header), Create(timeline), FadeIn(tracker), Create(graph))

        palette = [GREEN, ORANGE, PURPLE, RED, YELLOW]
        inactive_style = dict(width=1.6, opacity=0.08)

        for snapshot in snapshots:
            target_point = timeline.number_to_point(snapshot.time_index)
            label = Tex(f"Snapshot {snapshot.time_index + 1}").next_to(graph, UP)
            stats = MathTex(f"{len(snapshot.edges)}\\text{{ edges}}").next_to(
                label, DOWN, buff=0.1
            )

            self.play(
                tracker.animate.move_to(target_point),
                Write(label),
                FadeIn(stats),
                run_time=0.6,
            )

            active_anims = self._style_edges(
                edge_lines, snapshot.edges, palette, stroke_kwargs=dict(width=4, opacity=0.9)
            )
            if active_anims:
                self.play(*active_anims, run_time=0.7)

            self.wait(0.3)

            reset_anims = self._style_edges(
                edge_lines, snapshot.edges, [WHITE], stroke_kwargs=inactive_style
            )
            if reset_anims:
                self.play(*reset_anims, run_time=0.4)

            self.play(FadeOut(label), FadeOut(stats), run_time=0.2)

        self.wait(1)

    def _create_graph(self, nodes: List[int]) -> Graph:
        layout = self._circle_layout(nodes)
        graph = Graph(
            vertices=nodes,
            edges=[],
            layout=layout,
            labels=True,
            vertex_config={"radius": 0.16, "fill_color": BLUE},
        )
        return graph

    def _circle_layout(self, nodes: Iterable[int]) -> Dict[int, Tuple[float, float, float]]:
        radius = 3.5
        layout: Dict[int, Tuple[float, float, float]] = {}
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            layout[node] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
                0,
            )
        return layout

    def _build_edge_layer(
        self, graph: Graph, snapshots: Sequence[TemporalSnapshot]
    ) -> Tuple[Dict[Tuple[int, int], Line], VGroup]:
        centers = {node: graph.vertices[node].get_center() for node in graph.vertices}
        edge_lines: Dict[Tuple[int, int], Line] = {}

        for snapshot in snapshots:
            for u, v in snapshot.edges:
                if u == v:
                    continue
                key = tuple(sorted((u, v)))
                if key not in edge_lines:
                    line = Line(
                        centers[key[0]],
                        centers[key[1]],
                        stroke_color=WHITE,
                        stroke_opacity=0.08,
                        stroke_width=1.6,
                    )
                    edge_lines[key] = line

        edge_group = VGroup(*edge_lines.values())
        return edge_lines, edge_group

    def _style_edges(
        self,
        edge_lines: Dict[Tuple[int, int], Line],
        edges: Iterable[Tuple[int, int]],
        palette: Sequence,
        stroke_kwargs: Dict[str, float],
    ):
        animations = []
        for u, v in edges:
            key = tuple(sorted((u, v)))
            line = edge_lines.get(key)
            if not line:
                continue
            color = random.choice(palette) if palette else WHITE
            animations.append(line.animate.set_stroke(color=color, **stroke_kwargs))
        return animations


if __name__ == "__main__":
    from manim import config

    config.background_color = "#0f172a"
    TemporalGraphScene().render()
