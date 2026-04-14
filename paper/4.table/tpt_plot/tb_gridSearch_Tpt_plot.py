from halib import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


OUT_DIR = "./paper/4.table/tpt_plot"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("./paper/4.table/tpt_plot/tb_gridsearch_Tpt.csv", sep=";", encoding="utf-8")


pio.templates.default = "plotly_white"
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

# ── Load & split ──────────────────────────────────────────────────────────────
baseline = df[df["window_size"].isna()].iloc[0]
grid = df[df["window_size"].notna()].copy()
grid["window_size"] = grid["window_size"].astype(int)
grid["persist_thres"] = grid["persist_thres"].astype(float)

recall_base = baseline["metric_recall (tpr)"]
far_base = baseline["metric_fpr (false alarm rate)"]

grid_agg = (
    grid.groupby(["window_size", "persist_thres"])
    .agg(
        Recall=("metric_recall (tpr)", "mean"),
        FAR=("metric_fpr (false alarm rate)", "mean"),
    )
    .reset_index()
)

window_sizes = sorted(grid_agg["window_size"].unique())

# ── Pivots ────────────────────────────────────────────────────────────────────
recall_pivot = grid_agg.pivot(
    index="persist_thres", columns="window_size", values="Recall"
).sort_index(ascending=False)
far_pivot = grid_agg.pivot(
    index="persist_thres", columns="window_size", values="FAR"
).sort_index(ascending=False)
xs = [str(c) for c in recall_pivot.columns]
ys = [str(r) for r in recall_pivot.index]

# ── 1. Heatmap: Recall (tight range for color contrast) ──────────────────────
valid = recall_pivot.values[recall_pivot.values > 0]
fig1 = go.Figure(
    go.Heatmap(
        z=recall_pivot.values,
        x=xs,
        y=ys,
        colorscale="RdYlGn",
        zmin=valid.min() - 0.001,
        zmax=valid.max() + 0.001,
        colorbar=dict(title="Recall", tickformat=".4f"),
        hovertemplate="win=%{x}  thr=%{y}<br>Recall=%{z:.4f}<extra></extra>",
    )
)
fig1.update_layout(
    title=dict(text=f"Recall  (Baseline={recall_base:.4f})", x=0.5, font=dict(size=16)),
    xaxis=dict(title="window_size"),
    yaxis=dict(title="persist_thres"),
    margin=dict(l=90, r=110, t=70, b=60),
)
fig1.write_image(f"{OUT_DIR}/heatmap_recall.png")

# ── 2. Heatmap: FAR ───────────────────────────────────────────────────────────
fig2 = go.Figure(
    go.Heatmap(
        z=far_pivot.values,
        x=xs,
        y=ys,
        colorscale="RdYlGn_r",
        colorbar=dict(title="FAR", tickformat=".5f"),
        hovertemplate="win=%{x}  thr=%{y}<br>FAR=%{z:.5f}<extra></extra>",
    )
)
fig2.update_layout(
    title=dict(text=f"FAR  (Baseline={far_base:.5f})", x=0.5, font=dict(size=16)),
    xaxis=dict(title="window_size"),
    yaxis=dict(title="persist_thres"),
    margin=dict(l=90, r=110, t=70, b=60),
)
fig2.write_image(f"{OUT_DIR}/heatmap_far.png")

# ── 3. Scatter: Recall vs FAR ─────────────────────────────────────────────────
fig3 = go.Figure()
fig3.add_shape(
    type="rect",  # "better zone"
    x0=grid_agg["FAR"].min() * 0.995,
    x1=far_base,
    y0=recall_base,
    y1=grid_agg["Recall"].max() * 1.0004,
    fillcolor="rgba(0,200,100,0.12)",
    line=dict(width=0),
    layer="below",
)
fig3.add_annotation(
    x=far_base * 0.9985,
    y=recall_base * 1.0001,
    text="Better zone",
    showarrow=False,
    font=dict(size=10, color="green"),
    xanchor="right",
    yanchor="bottom",
)

for ws, col in zip(window_sizes, COLORS):
    sub = grid_agg[grid_agg["window_size"] == ws]
    fig3.add_trace(
        go.Scatter(
            x=sub["FAR"],
            y=sub["Recall"],
            mode="markers",
            name=f"w={ws}",
            marker=dict(
                color=col,
                size=sub["persist_thres"] * 26 + 5,
                opacity=0.78,
                line=dict(width=0.8, color="white"),
            ),
            customdata=sub["persist_thres"].values,
            hovertemplate=f"w={ws}, thr=%{{customdata:.2f}}<br>"
            f"Recall=%{{y:.4f}}, FAR=%{{x:.5f}}<extra></extra>",
        )
    )
fig3.add_trace(
    go.Scatter(
        x=[far_base],
        y=[recall_base],
        mode="markers",
        name="Baseline A",
        marker=dict(color="red", size=14, symbol="star"),
    )
)
fig3.add_hline(y=recall_base, line=dict(color="red", dash="dot", width=1), opacity=0.45)
fig3.add_vline(x=far_base, line=dict(color="red", dash="dot", width=1), opacity=0.45)
fig3.update_layout(
    title=dict(text="Recall vs FAR — Grid Search", x=0.5, font=dict(size=16)),
    xaxis=dict(title="FAR", tickformat=".5f"),
    yaxis=dict(title="Recall", tickformat=".4f"),
    legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
    margin=dict(l=70, r=140, t=70, b=60),
)
fig3.write_image(f"{OUT_DIR}/scatter_recall_far.png")
# ── 4. Dual-panel line chart ──────────────────────────────────────────────────
fig4 = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=["Recall vs persist_thres", "FAR vs persist_thres"],
)
for ws, col in zip(window_sizes, COLORS):
    sub = grid_agg[grid_agg["window_size"] == ws].sort_values("persist_thres")
    fig4.add_trace(
        go.Scatter(
            x=sub["persist_thres"],
            y=sub["Recall"],
            mode="lines+markers",
            name=f"w={ws}",
            legendgroup=str(ws),
            line=dict(color=col, width=2),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )
    fig4.add_trace(
        go.Scatter(
            x=sub["persist_thres"],
            y=sub["FAR"],
            mode="lines+markers",
            name=f"w={ws}",
            legendgroup=str(ws),
            line=dict(color=col, width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

fig4.add_hline(
    y=recall_base,
    row=1,
    line=dict(color="red", dash="dash", width=1.5),
    annotation_text=f"Baseline {recall_base:.4f}",
    annotation_position="bottom right",
    annotation_font=dict(size=10, color="red"),
)
fig4.add_hline(
    y=far_base,
    row=2,
    line=dict(color="red", dash="dash", width=1.5),
    annotation_text=f"Baseline {far_base:.5f}",
    annotation_position="top right",
    annotation_font=dict(size=10, color="red"),
)
fig4.update_xaxes(title_text="persist_thres", row=2)
fig4.update_yaxes(title_text="Recall", tickformat=".4f", row=1)
fig4.update_yaxes(title_text="FAR", tickformat=".5f", row=2)
fig4.update_layout(
    title=dict(text="Recall & FAR vs persist_thres", x=0.5, font=dict(size=16)),
    legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
    margin=dict(l=80, r=120, t=80, b=60),
    height=600,
)
fig4.write_image(f"{OUT_DIR}/__lines_recall_far.pdf")
fig4.write_html(f"{OUT_DIR}/__lines_recall_far.html")
