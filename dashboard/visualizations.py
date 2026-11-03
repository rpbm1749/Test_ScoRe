"""Visualization utilities for ScoRe Dashboard.

Provides Mermaid graph generators, Plotly charts for token/cost analysis,
and styled HTML helpers for step trajectory inspection.
"""

from __future__ import annotations

import json
from typing import Any
import plotly.graph_objects as go
import plotly.express as px


def _sanitize_mermaid(text: str) -> str:
    """Sanitize strings for Mermaid node labels."""
    if not text:
        return ""
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "#quot;")
        .replace("'", "&#39;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )
    return text


def build_mermaid_graph(steps: list[dict[str, Any]], windows: list[dict[str, Any]]) -> str:
    """Generate a Mermaid diagram definition representing the multi-turn Planner/Executor trajectory."""
    if not steps:
        return "graph TD\n  empty[No steps recorded]"

    # Map step number to step record for quick lookup
    step_dict = {s["step"]: s for s in steps}

    lines = [
        "%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3F2FD', 'edgeLabelBackground':'#FFFFFF'}}}%%",
        "graph TD",
    ]

    # Style classes
    lines.extend([
        "  classDef plannerNode fill:#BBDEFB,stroke:#1976D2,stroke-width:2px,color:#0D47A1;",
        "  classDef executorNode fill:#C8E6C9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;",
        "  classDef errorNode fill:#FFCDD2,stroke:#D32F2F,stroke-width:2px,color:#B71C1C;",
        "  classDef handbackNode fill:#FFE082,stroke:#FFA000,stroke-width:2px,color:#E65100,stroke-dasharray: 5 5;",
        "  classDef finishNode fill:#E1BEE7,stroke:#8E24AA,stroke-width:3px,color:#4A148C;",
    ])

    prev_node_id: str | None = None

    if windows:
        for w in windows:
            w_num = w.get("window", 1)
            p_step = w.get("planner_step")
            e_steps = w.get("executor_steps", [])
            hb_reason = w.get("handback_reason", "")

            lines.append(f"  subgraph Window_{w_num} [\"🪟 Window {w_num}\"]")

            # Planner node
            if p_step is not None and p_step in step_dict:
                s_rec = step_dict[p_step]
                first_line = (s_rec.get("code") or "").strip().splitlines()
                raw_snip = (first_line[0][:35] + "...") if first_line else "no code"
                code_snip = _sanitize_mermaid(raw_snip)
                is_err = s_rec.get("error", False)
                p_id = f"s_{p_step}"
                lines.append(f"    {p_id}[\"🧠 Step {p_step} (Planner)<br/><code>{code_snip}</code>\"]")
                lines.append(f"    class {p_id} {'errorNode' if is_err else 'plannerNode'};")

                if prev_node_id:
                    lines.append(f"    {prev_node_id} --> {p_id}")
                prev_node_id = p_id

            # Executor nodes
            for es in e_steps:
                if es in step_dict:
                    s_rec = step_dict[es]
                    first_line = (s_rec.get("code") or "").strip().splitlines()
                    raw_snip = (first_line[0][:35] + "...") if first_line else "no code"
                    code_snip = _sanitize_mermaid(raw_snip)
                    is_err = s_rec.get("error", False)
                    e_id = f"s_{es}"
                    lines.append(f"    {e_id}[\"⚡ Step {es} (Executor)<br/><code>{code_snip}</code>\"]")
                    lines.append(f"    class {e_id} {'errorNode' if is_err else 'executorNode'};")

                    if prev_node_id:
                        lines.append(f"    {prev_node_id} --> {e_id}")
                    prev_node_id = e_id

            # Handback node
            if hb_reason:
                hb_clean = _sanitize_mermaid(hb_reason[:60])
                hb_id = f"hb_{w_num}"
                lines.append(f"    {hb_id}[\"↩️ Handback: {hb_clean}\"]")
                lines.append(f"    class {hb_id} handbackNode;")
                if prev_node_id:
                    lines.append(f"    {prev_node_id} -.-> {hb_id}")
                prev_node_id = hb_id

            lines.append("  end")
    else:
        # Fallback linear rendering if no window grouping
        for s in steps:
            s_num = s["step"]
            actor = s.get("actor", "agent")
            first_line = (s.get("code") or "").strip().splitlines()
            raw_snip = (first_line[0][:35] + "...") if first_line else "no code"
            code_snip = _sanitize_mermaid(raw_snip)
            is_err = s.get("error", False)
            s_id = f"s_{s_num}"
            node_class = "errorNode" if is_err else ("plannerNode" if actor == "planner" else "executorNode")
            lines.append(f"  {s_id}[\"Step {s_num} ({actor})<br/><code>{code_snip}</code>\"]")
            lines.append(f"  class {s_id} {node_class};")
            if prev_node_id:
                lines.append(f"  {prev_node_id} --> {s_id}")
            prev_node_id = s_id

    return "\n".join(lines)


def render_mermaid_html(mermaid_code: str, height: int = 500) -> str:
    """Render Mermaid syntax inside an isolated responsive HTML iframe."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 10px;
                background-color: transparent;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 15px;
                overflow: auto;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{mermaid_code}
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
        </script>
    </body>
    </html>
    """


def create_token_comparison_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Generate a Plotly bar chart comparing frontier vs cheap tokens per task."""
    tasks = [r["task_id"] for r in results]
    frontier_tokens = [r.get("frontier_prompt_tokens", 0) + r.get("frontier_completion_tokens", 0) for r in results]
    cheap_tokens = [r.get("cheap_prompt_tokens", 0) + r.get("cheap_completion_tokens", 0) for r in results]

    fig = go.Figure(data=[
        go.Bar(name="Frontier (Smart/Costly)", x=tasks, y=frontier_tokens, marker_color="#1976D2"),
        go.Bar(name="Cheap (Executor)", x=tasks, y=cheap_tokens, marker_color="#43A047"),
    ])
    fig.update_layout(
        barmode="stack",
        title="Token Consumption per Task (Frontier vs Cheap)",
        xaxis_title="Task ID",
        yaxis_title="Total Tokens",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
    )
    return fig


def create_step_breakdown_chart(steps_by_actor: dict[str, int]) -> go.Figure:
    """Generate a donut chart showing the percentage of steps handled by Planner vs Executor."""
    labels = list(steps_by_actor.keys())
    values = list(steps_by_actor.values())
    colors = ["#1976D2", "#43A047"]

    fig = go.Figure(data=[go.Pie(
        labels=[k.capitalize() for k in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent+value",
    )])
    fig.update_layout(
        title="Steps by Model Actor",
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
