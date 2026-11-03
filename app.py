"""ScoRe Experiment & Trajectory Dashboard.

A rich Streamlit dashboard to explore, analyze, and visualize results from the runs/ folder.
Visualizes step flows, model delegations, token metrics, and execution trajectories.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Any

from dashboard.data_loader import (
    get_available_runs,
    load_run_results,
    load_run_config,
    load_task_trajectory,
    load_task_specs,
)
from dashboard.visualizations import (
    build_mermaid_graph,
    render_mermaid_html,
    create_token_comparison_chart,
    create_step_breakdown_chart,
)

# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ScoRe Agent Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #1976D2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .badge-planner {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #90CAF9;
    }
    .badge-executor {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #A5D6A7;
    }
    .badge-handback {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #FFCC80;
    }
    .task-instruction-box {
        background-color: #F0F4F8;
        border: 1px solid #D0DCE5;
        border-radius: 6px;
        padding: 16px;
        font-size: 15px;
        line-height: 1.5;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Run & Task Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🎯 ScoRe Dashboard")
st.sidebar.markdown("*Frontier-Planner & Cheap-Executor Trajectory Explorer*")

available_runs = get_available_runs()

if not available_runs:
    st.warning("⚠️ No completed runs found in the `runs/` folder. Execute an experiment first (e.g. `python run.py --mode score --k 3`).")
    st.stop()

selected_run = st.sidebar.selectbox("📁 Select Run", options=available_runs, index=0)

# Load selected run data
results_data = load_run_results(selected_run)
config_data = load_run_config(selected_run)

# Sidebar metadata overview
if config_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Run Config")
    models_cfg = config_data.get("models", {})
    score_cfg = config_data.get("score", {})
    st.sidebar.markdown(f"**Mode:** `{config_data.get('run', {}).get('mode', 'score')}`")
    st.sidebar.markdown(f"**Planner:** `{models_cfg.get('frontier', {}).get('model', 'N/A')}`")
    st.sidebar.markdown(f"**Executor:** `{models_cfg.get('cheap', {}).get('model', 'N/A')}`")
    st.sidebar.markdown(f"**Window (k):** `{score_cfg.get('k', 'N/A')}`")
    st.sidebar.markdown(f"**View:** `{score_cfg.get('planner_view', 'notes')}`")

# Sidebar Task Selector
selected_task_id = None
if results_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Task Selector")
    task_options = [r["task_id"] for r in results_data]
    task_labels = {
        r["task_id"]: f"{'✅' if r.get('success') else '❌'} {r['task_id']} ({r.get('pass_percentage', 0):.0f}%)"
        for r in results_data
    }
    selected_task_id = st.sidebar.selectbox(
        "Choose Task",
        options=task_options,
        format_func=lambda tid: task_labels.get(tid, tid),
        index=0,
    )

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Main Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Run Overview & Metrics",
    "🔬 Task Trajectory & Step Graph",
    "🤝 ScoRe Delegation & Handoffs",
    "⚙️ Configuration",
])

# -----------------------------------------------------------------------------
# TAB 1: Run Overview
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Run Summary: `{selected_run}`")

    if not results_data:
        st.info("No task records found in `results.jsonl` for this run.")
    else:
        # High-level Metrics Row
        total_tasks = len(results_data)
        completed_tasks = sum(1 for r in results_data if r.get("success", False))
        avg_pass = sum(r.get("pass_percentage", 0) for r in results_data) / max(1, total_tasks)
        avg_steps = sum(r.get("env_steps", 0) for r in results_data) / max(1, total_tasks)

        tot_frontier_tokens = sum(r.get("frontier_prompt_tokens", 0) + r.get("frontier_completion_tokens", 0) for r in results_data)
        tot_cheap_tokens = sum(r.get("cheap_prompt_tokens", 0) + r.get("cheap_completion_tokens", 0) for r in results_data)
        tot_frontier_calls = sum(r.get("frontier_calls", 0) for r in results_data)
        tot_cheap_calls = sum(r.get("cheap_calls", 0) for r in results_data)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tasks Completed", f"{completed_tasks} / {total_tasks}", f"{(completed_tasks/max(1, total_tasks))*100:.1f}%")
        m2.metric("Avg Pass Rate", f"{avg_pass:.1f}%")
        m3.metric("Avg Steps / Task", f"{avg_steps:.1f}")
        m4.metric("Frontier Calls", f"{tot_frontier_calls:,}", f"{tot_frontier_tokens:,} tokens")
        m5.metric("Cheap Calls", f"{tot_cheap_calls:,}", f"{tot_cheap_tokens:,} tokens")

        st.markdown("---")

        # Visual Charts
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            fig_tokens = create_token_comparison_chart(results_data)
            st.plotly_chart(fig_tokens, use_container_width=True)

        with col_c2:
            actor_steps = {"planner": 0, "executor": 0}
            for r in results_data:
                s_by_a = r.get("steps_by_actor", {})
                actor_steps["planner"] += s_by_a.get("planner", 0)
                actor_steps["executor"] += s_by_a.get("executor", 0)
            fig_pie = create_step_breakdown_chart(actor_steps)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tabular View
        st.markdown("### 📋 Results Table")
        df_display = []
        for r in results_data:
            df_display.append({
                "Task ID": r.get("task_id"),
                "Success": "✅ Yes" if r.get("success") else "❌ No",
                "Pass %": f"{r.get('pass_percentage', 0):.1f}%",
                "Steps": r.get("env_steps", 0),
                "Planner Steps": r.get("steps_by_actor", {}).get("planner", 0),
                "Executor Steps": r.get("steps_by_actor", {}).get("executor", 0),
                "Wall Time (s)": f"{r.get('wall_time_s', 0):.1f}",
                "Frontier Tokens": r.get("frontier_prompt_tokens", 0) + r.get("frontier_completion_tokens", 0),
                "Cheap Tokens": r.get("cheap_prompt_tokens", 0) + r.get("cheap_completion_tokens", 0),
                "Rejections": r.get("rejections", 0),
                "End Reason": r.get("end_reason", ""),
            })
        st.dataframe(pd.DataFrame(df_display), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Task Trajectory & Step Graph
# -----------------------------------------------------------------------------
with tab2:
    if not results_data or not selected_task_id:
        st.info("No tasks to display.")
    else:
        # Load task data
        traj_data = load_task_trajectory(selected_run, selected_task_id)
        spec_data = load_task_specs(selected_task_id)

        if not traj_data:
            st.error(f"Could not load trajectory for task `{selected_task_id}`. Check `{selected_task_id}.json`.")
        else:
            # 1. Header Information
            instruction = spec_data.get("instruction") or "(Instruction not found in local specs.json)"
            supervisor = spec_data.get("supervisor", {})

            st.markdown(f"### 🎯 Task: `{selected_task_id}`")
            st.markdown(f"""
            <div class="task-instruction-box">
                <b>User Instruction:</b><br/>
                {instruction}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("👤 Task Supervisor & Context Details", expanded=False):
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.write(f"**Name:** {supervisor.get('first_name', '')} {supervisor.get('last_name', '')}")
                col_s2.write(f"**Email:** `{supervisor.get('email', '')}`")
                col_s3.write(f"**Phone:** `{supervisor.get('phone_number', '')}`")

            # 2. Visual Step Graph (Flowchart)
            st.markdown("### 🗺️ Visual Step Execution Graph")
            st.caption("Illustrates the orchestration between Planner turns, Executor windows, and Handback events.")

            mermaid_code = build_mermaid_graph(traj_data.get("steps", []), traj_data.get("windows", []))
            html_graph = render_mermaid_html(mermaid_code, height=520)
            st.components.v1.html(html_graph, height=540, scrolling=True)

            # 3. Detailed Step-by-Step Inspector
            st.markdown("### 🪜 Step-by-Step Execution Log")
            steps = traj_data.get("steps", [])

            for s in steps:
                s_num = s.get("step")
                actor = s.get("actor", "agent")
                window_num = s.get("window", "-")
                is_err = s.get("error", False)
                first_code = (s.get("code") or "").strip().splitlines()
                preview = first_code[0][:80] if first_code else "no code"

                # Status icon & actor badge
                status_icon = "❌ ERROR" if is_err else "✅ ok"
                actor_badge = f"🧠 Planner (w{window_num})" if actor == "planner" else f"⚡ Executor (w{window_num})"

                with st.expander(f"Step {s_num:>2} | {actor_badge} | {status_icon} | `{preview}`"):
                    # Thought
                    if s.get("thought"):
                        st.markdown("**💭 Thought / Reasoning:**")
                        st.info(s["thought"])

                    # Code
                    st.markdown("**💻 Python Code:**")
                    st.code(s.get("code") or "# (empty code block)", language="python")

                    # Output
                    st.markdown("**🖥️ Output:**")
                    output_text = s.get("output") or "(no output)"
                    if is_err:
                        st.error(output_text)
                    else:
                        st.text_area("Output", value=output_text, height=140, disabled=True, key=f"out_{s_num}")

            # Rejections review (if any)
            rejections = traj_data.get("rejections", [])
            if rejections:
                st.markdown("### ⚠️ Pre-execution Checker Rejections")
                st.caption("Steps that were halted and corrected before running in AppWorld.")
                for idx, r in enumerate(rejections, 1):
                    with st.expander(f"Rejection #{idx} [{r.get('role')}] in Window {r.get('window')}"):
                        st.warning(f"**Problem Caught:**\n{r.get('problem')}")
                        st.code(r.get("code"), language="python")


# -----------------------------------------------------------------------------
# TAB 3: ScoRe Delegation & Handoffs
# -----------------------------------------------------------------------------
with tab3:
    if not results_data:
        st.info("No run data available.")
    else:
        st.subheader("Window-by-Window Delegation Inspector")
        st.caption("Study how the Planner directs the Executor, self-monitors with notes, and why control gets handed back.")

        traj_data = load_task_trajectory(selected_run, selected_task_id)
        if not traj_data:
            st.error("No trajectory loaded.")
        else:
            windows = traj_data.get("windows", [])
            if not windows:
                st.info("No window records found. This mode may have been run without ScoRe windowing.")
            else:
                for w in windows:
                    w_num = w.get("window")
                    p_step = w.get("planner_step")
                    e_steps = w.get("executor_steps", [])
                    hb_reason = w.get("handback_reason") or "N/A"

                    with st.container():
                        st.markdown(f"#### 🪟 Window {w_num}")
                        col_w1, col_w2, col_w3 = st.columns([1, 1, 2])
                        col_w1.markdown(f"**Planner Step:** `{p_step if p_step is not None else 'None'}`")
                        col_w2.markdown(f"**Executor Steps:** `{e_steps}` ({len(e_steps)} steps)")
                        col_w3.markdown(f"<span class='badge-handback'>Handback: {hb_reason}</span>", unsafe_allow_html=True)

                        c_inst, c_notes = st.columns(2)
                        with c_inst:
                            st.markdown("**📝 Planner Instructions to Executor:**")
                            inst_text = w.get("instructions") or "*(No instructions provided)*"
                            st.code(inst_text, language="markdown")

                        with c_notes:
                            st.markdown("**📓 Planner Progress Notes (Notes View):**")
                            notes_text = w.get("notes") or "*(No progress notes)*"
                            st.info(notes_text)

                        st.markdown("---")


# -----------------------------------------------------------------------------
# TAB 4: Configuration
# -----------------------------------------------------------------------------
with tab4:
    st.subheader(f"⚙️ Configuration for `{selected_run}`")
    st.caption("Stored in `config_used.yaml` for reproducibility.")
    if config_data:
        st.json(config_data)
    else:
        st.warning("No `config_used.yaml` found in this run directory.")
