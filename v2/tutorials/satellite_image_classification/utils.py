"""
Utility functions for satellite image classification pipeline.

Includes:
- t-SNE feature visualization
- W&B integration helpers
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import wandb
from sklearn.manifold import TSNE
from flyteplugins.wandb import get_wandb_run

# Fixed color per class so colors are consistent across all t-SNE updates
CLASS_COLORS = {
    "Annual_Crop":            "#1f77b4",
    "Forest":                 "#2ca02c",
    "Herbaceous_Vegetation":  "#98df8a",
    "Industrial":             "#d62728",
    "Pasture":                "#ffbb78",
    "Permanent_Crop":         "#ff7f0e",
    "Residential":            "#9467bd",
    "River":                  "#17becf",
    "SeaLake":                "#aec7e8",
    "Glacier":                "#e377c2",
}


def compute_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_components: int = 3,
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute t-SNE projection of features.
    
    Args:
        features: Feature tensor of shape (n_samples, feature_dim)
        labels: Label tensor of shape (n_samples,)
        n_components: Number of t-SNE dimensions (2 or 3)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        
    Returns:
        t-SNE coordinates of shape (n_samples, n_components)
    """
    if len(features) == 0:
        return np.array([])

    # Convert to numpy
    features_np = features.numpy() if isinstance(features, torch.Tensor) else features
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # Adjust perplexity for small datasets
    n_samples = len(features_np)
    actual_perplexity = min(perplexity, (n_samples - 1) // 3)

    # Compute t-SNE — pca init gives a stable global orientation across epochs
    tsne = TSNE(
        n_components=n_components,
        perplexity=actual_perplexity,
        random_state=random_state,
        init="pca",
        max_iter=1000,
        verbose=0,
    )
    tsne_features = tsne.fit_transform(features_np)

    return tsne_features


def create_2d_tsne_plot(
    tsne_features: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str = "t-SNE Feature Visualization (2D)",
) -> go.Figure:
    """
    Create 2D t-SNE plot with Plotly.
    
    Args:
        tsne_features: t-SNE coordinates of shape (n_samples, 2)
        labels: Class labels
        class_names: List of class names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Create dataframe-like data
    df_data = {
        "x": tsne_features[:, 0],
        "y": tsne_features[:, 1],
        "label": [class_names[l] for l in labels],
    }

    fig = px.scatter(
        df_data,
        x="x",
        y="y",
        color="label",
        color_discrete_map=CLASS_COLORS,
        title=title,
        hover_name="label",
        width=800,
        height=800,
    )
    fig.update_layout(
        font=dict(size=12),
        hovermode="closest",
    )
    return fig


def _balanced_sample(
    features: torch.Tensor,
    labels: torch.Tensor,
    samples_per_class: int = 50,
) -> tuple:
    """Return at most `samples_per_class` examples per unique label."""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else features
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    selected = []
    rng = np.random.default_rng(42)
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        if len(idx) > samples_per_class:
            idx = rng.choice(idx, samples_per_class, replace=False)
        selected.append(idx)

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return torch.tensor(features_np[selected]), torch.tensor(labels_np[selected])


def log_tsne_to_wandb(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list,
    split: str = "validation",
    epoch: int = 0,
    n_components: int = 2,
):
    if len(features) == 0:
        return

    features, labels = _balanced_sample(features, labels, samples_per_class=50)
    tsne_features = compute_tsne(features, labels, n_components=n_components)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    fig = create_2d_tsne_plot(
        tsne_features,
        labels_np,
        class_names,
        title=f"t-SNE {split.title()} (Epoch {epoch})",
    )
    get_wandb_run().log({f"tsne/{split}/plot": wandb.Plotly(fig), "epoch": epoch})


def get_model_size(model: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_trainable_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
