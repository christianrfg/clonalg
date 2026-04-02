import sys
import time
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, "src")
from clonalg import CLONALGConfig, run  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CLONALG",
    page_icon="🧬",
    layout="wide",
)

# ── Objective functions ───────────────────────────────────────────────────────

OBJECTIVES: dict[str, tuple[str, Callable]] = {
    "Sphere": (
        r"$f(x) = \sum x_i^2$ — convex, unimodal. Optimum at origin.",
        lambda x: float(np.sum(x**2)),
    ),
    "Rastrigin": (
        r"$f(x) = 10n + \sum [x_i^2 - 10\cos(2\pi x_i)]$ — highly multimodal.",
        lambda x: float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))),
    ),
    "Rosenbrock": (
        r"$f(x) = \sum [100(x_{i+1}-x_i^2)^2 + (1-x_i)^2]$ — narrow curved valley.",
        lambda x: float(
            sum(
                100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            ),
        ),
    ),
}

# ── Sidebar — configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 CLONALG")
    st.caption("CLONal selection ALGorithm")
    st.divider()

    st.subheader("Objective Function")
    obj_name = st.selectbox(
        "Function",
        options=list(OBJECTIVES.keys()),
        label_visibility="collapsed",
    )
    obj_desc, objective_fn = OBJECTIVES[obj_name]
    st.markdown(obj_desc)

    st.divider()
    st.subheader("Problem")
    problem_size = st.slider(
        "Problem size (dimensions)",
        min_value=2,
        max_value=10,
        value=3,
    )
    b_lo = st.number_input("Lower bound", value=-5.0, step=1.0)
    b_up = st.number_input("Upper bound", value=5.0, step=1.0)

    st.divider()
    st.subheader("Algorithm")
    population_size = st.slider(
        "Population size",
        min_value=20,
        max_value=500,
        value=200,
        step=10,
    )
    selection_size = st.slider(
        "Selection size",
        min_value=2,
        max_value=100,
        value=20,
    )
    random_cells_num = st.slider(
        "Random cells per iteration",
        min_value=5,
        max_value=100,
        value=30,
    )
    clone_rate = st.slider("Clone rate", min_value=5, max_value=100, value=50)
    mutation_rate = st.slider(
        "Mutation rate",
        min_value=0.01,
        max_value=1.0,
        value=0.3,
        step=0.01,
    )
    max_iterations = st.slider(
        "Max iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )

    st.divider()
    st.subheader("Display")
    update_every = st.slider(
        "Update plot every N iterations",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )
    delay = st.slider(
        "Delay per update (s)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )

    st.divider()
    run_button = st.button("▶  Run", type="primary", width="stretch")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("CLONALG — Convergence Explorer")
st.markdown(
    "Adjust the parameters in the sidebar and press **Run** "
    "to watch the algorithm converge in real time.",
)

col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    chart_placeholder = st.empty()

with col_metrics:
    st.markdown("### Live metrics")
    metric_iteration = st.empty()
    metric_best = st.empty()
    metric_mean = st.empty()
    st.divider()
    progress_bar = st.progress(0, text="Waiting to start…")

result_placeholder = st.empty()


def make_figure(
    iters: list[int],
    best_history: list[float],
    mean_history: list[float],
    current: int,
    total: int,
) -> go.Figure:
    fig = go.Figure()

    # Shaded band
    fig.add_trace(
        go.Scatter(
            x=iters + iters[::-1],
            y=mean_history + best_history[::-1],
            fill="toself",
            fillcolor="rgba(132, 94, 194, 0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=mean_history,
            mode="lines",
            name="Top-5 mean",
            line=dict(color="#845ec2", width=1.5, dash="dot"),
            hovertemplate="Iter %{x}<br>Mean: %{y:.6f}<extra></extra>",
        )
    )

    # Best line
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=best_history,
            mode="lines",
            name="Best",
            line=dict(color="#00c9a7", width=2.5),
            hovertemplate="Iter %{x}<br>Best: %{y:.8f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{obj_name} — iteration {current} / {total}",
            font=dict(size=14, color="white"),
            x=0.0,
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#161b22",
        font=dict(color="#cccccc", size=11),
        xaxis=dict(
            title="Iteration",
            range=[0, total],
            gridcolor="#1e2530",
            zerolinecolor="#1e2530",
        ),
        yaxis=dict(
            title="Affinity (log scale)",
            type="log",
            gridcolor="#1e2530",
            zerolinecolor="#1e2530",
        ),
        legend=dict(
            bgcolor="#1a1d27",
            bordercolor="#333333",
            borderwidth=1,
            orientation="h",
            y=1.08,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=50),
        height=460,
    )
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

if run_button:
    if b_lo >= b_up:
        st.error("Lower bound must be less than upper bound.")
        st.stop()

    if selection_size >= population_size:
        st.error("Selection size must be less than population size.")
        st.stop()

    config = CLONALGConfig(
        problem_size=problem_size,
        b_lo=b_lo,
        b_up=b_up,
        population_size=population_size,
        selection_size=selection_size,
        random_cells_num=random_cells_num,
        clone_rate=clone_rate,
        mutation_rate=mutation_rate,
        max_iterations=max_iterations,
    )

    best_history: list[float] = []
    mean_history: list[float] = []

    def on_iteration(i: int, top5: list) -> None:
        best_history.append(top5[0][1])
        mean_history.append(float(np.mean([ind[1] for ind in top5])))

        progress = (i + 1) / max_iterations
        progress_bar.progress(progress, text=f"Iteration {i + 1} / {max_iterations}")
        metric_iteration.metric("Iteration", f"{i + 1} / {max_iterations}")
        metric_best.metric("Best affinity", f"{best_history[-1]:.6f}")
        metric_mean.metric("Top-5 mean", f"{mean_history[-1]:.6f}")

        if i % update_every == 0 or i == max_iterations - 1:
            iters = list(range(len(best_history)))
            fig = make_figure(iters, best_history, mean_history, i + 1, max_iterations)
            chart_placeholder.plotly_chart(fig, width="stretch")
            if delay > 0:
                time.sleep(delay)

    result = run(objective_fn, config, on_iteration=on_iteration)

    progress_bar.progress(1.0, text="Done!")

    # Final chart (x-axis fully filled)
    iters = list(range(len(best_history)))
    final_fig = make_figure(
        iters,
        best_history,
        mean_history,
        max_iterations,
        max_iterations,
    )
    final_fig.add_annotation(
        x=len(best_history) - 1,
        y=result.best_affinity,
        text=f"  {result.best_affinity:.6f}",
        showarrow=False,
        font=dict(color="#00c9a7", size=11),
        xanchor="left",
    )
    final_fig.update_layout(
        title_text=f"{obj_name} — converged after {max_iterations} iterations",
    )
    chart_placeholder.plotly_chart(final_fig, width="stretch")

    with result_placeholder.container():
        st.divider()
        st.subheader("Result")
        res_cols = st.columns(len(result.best_solution) + 1)
        for dim, (col, val) in enumerate(zip(res_cols, result.best_solution)):
            col.metric(f"x[{dim}]", f"{val:.6f}")
        res_cols[-1].metric("f(x*)", f"{result.best_affinity:.8f}")
