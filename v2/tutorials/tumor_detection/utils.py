"""
Utility functions for brain tumor MRI classification pipeline.

Includes confusion matrix, per-class F1 bar chart, and loss/accuracy curve helpers
for rendering in the Union UI report panel.
"""

import numpy as np
import plotly.graph_objects as go
import torch


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]


def create_confusion_matrix_plot(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list = CLASS_NAMES,
) -> go.Figure:
    """Normalized confusion matrix heatmap (row = true class, col = predicted)."""
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        matrix[t][p] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = matrix / (row_sums + 1e-8)
    text = [[f"{norm[i][j]:.2f}<br>({matrix[i][j]})" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=norm,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        zmin=0,
        zmax=1,
    ))
    fig.update_layout(
        title="Confusion Matrix (normalized)",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=600,
        height=600,
    )
    return fig


def create_metrics_plots(history: list):
    """Return (loss_fig, acc_fig) from per-epoch history list."""
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    val_loss = [e["val_loss"] for e in history]
    val_acc = [e["val_acc"] for e in history]
    macro_f1 = [e.get("macro_f1", 0) for e in history]

    phase_boundary = next((e["epoch"] for e in history if e["phase"] == 2), None)

    def add_phase_line(fig):
        if phase_boundary is not None:
            fig.add_vline(
                x=phase_boundary,
                line_dash="dash",
                line_color="gray",
                annotation_text="Phase 2 start",
            )

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss"))
    loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss"))
    loss_fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
    add_phase_line(loss_fig)

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(
        x=epochs, y=val_acc, mode="lines+markers", name="Val Accuracy",
        line=dict(color="#2ca02c"),
    ))
    acc_fig.add_trace(go.Scatter(
        x=epochs, y=macro_f1, mode="lines+markers", name="Macro F1",
        line=dict(color="#ff7f0e"),
    ))
    acc_fig.update_layout(title="Accuracy & F1", xaxis_title="Epoch", yaxis_title="Score")
    add_phase_line(acc_fig)

    return loss_fig, acc_fig


def create_per_class_f1_plot(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list = CLASS_NAMES,
) -> go.Figure:
    """Bar chart of per-class F1 scores."""
    f1_scores = []
    for cls_idx in range(len(class_names)):
        tp = ((preds == cls_idx) & (targets == cls_idx)).sum()
        fp = ((preds == cls_idx) & (targets != cls_idx)).sum()
        fn = ((preds != cls_idx) & (targets == cls_idx)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_scores.append(float(2 * precision * recall / (precision + recall + 1e-8)))

    fig = go.Figure(go.Bar(
        x=class_names,
        y=f1_scores,
        marker_color=CLASS_COLORS,
        text=[f"{s:.3f}" for s in f1_scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="Per-Class F1 Score",
        xaxis_title="Class",
        yaxis_title="F1",
        yaxis_range=[0, 1.15],
        width=600,
        height=400,
    )
    return fig


def get_model_size(model: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_trainable_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
