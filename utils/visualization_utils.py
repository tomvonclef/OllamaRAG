"""Visualization utilities for OllamaRAG application."""
import pandas as pd
import umap
import plotly.express as px


def plot_embedding_space(embeddings, chunks, retrieved_indices=None):
    """Project embeddings to 2D and visualize with interactive Plotly scatter."""
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "text": chunks,
        "retrieved": ["Yes" if i in retrieved_indices else "No" for i in range(len(chunks))]
    })

    fig = px.scatter(
        df_plot,
        x="x", y="y",
        color="retrieved",
        hover_data={"text": True},
        title="📊 Embedding Space (UMAP Projection)",
        labels={"retrieved": "Retrieved Chunk"}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig
