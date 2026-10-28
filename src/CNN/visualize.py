"""Build a mouse-rotatable KNN emotion map from recognized EmoArt artworks."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from numpy.typing import NDArray
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import DataLoader

from . import param
from .data import EmoArtDataset, seed_everything
from .model import ArtEmotionCNN

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class VisualizationArguments:
    """Command-line options for generating and querying the emotion map.

    Parameters:
        image: Optional artwork whose position is interpolated into the map.
        neighbors: Number of neighbors used to build the graph and place a query.
        clusters: Number of spectral groups assigned from the KNN graph.
        output: Destination HTML file for the interactive plot.
        seed: Random seed used by the spectral projection and clustering.

    Returns:
        An immutable visualization configuration object.
    """

    image: Path | None
    neighbors: int
    clusters: int
    output: Path
    seed: int


def _extract_recognition(
    frame: pd.DataFrame,
    model: ArtEmotionCNN,
    emotion_to_index: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    """Collect the CNN's learned image embeddings and emotion probabilities.

    Parameters:
        frame: Known artwork rows used to build the KNN reference space.
        model: Trained emotion-recognition CNN.
        emotion_to_index: Mapping from recognized emotion labels to output
            positions.
        device: Torch device used for inference.
        batch_size: Number of images inferred per batch.

    Returns:
        Tuple of image embeddings and class-probability arrays in frame order.

    Raises:
        ValueError: If frame is empty or batch_size is not positive.
        OSError: If an artwork image cannot be opened.
    """
    if frame.empty:
        raise ValueError("At least one known artwork is needed to create a KNN reference space")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive; received {batch_size}")
    dataset = EmoArtDataset(frame, emotion_to_index)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    embedding_batches: list[FloatArray] = []
    probability_batches: list[FloatArray] = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits, embeddings = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            embedding_batches.append(embeddings.cpu().numpy().astype(np.float32))
            probability_batches.append(probabilities.cpu().numpy().astype(np.float32))
    if not embedding_batches:
        raise ValueError("No image batches were produced for the known artworks")
    return np.concatenate(embedding_batches), np.concatenate(probability_batches)


def _normalize_rows(values: FloatArray) -> FloatArray:
    """Normalize each representation vector to unit L2 length.

    Parameters:
        values: Two-dimensional array of image or emotion vectors.

    Returns:
        A float32 array with each row normalized, leaving zero rows as zeros.

    Raises:
        ValueError: If values is not a non-empty two-dimensional array.
    """
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Expected a non-empty matrix; got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (values / norms).astype(np.float32)


def _joint_representation(
    image_embeddings: FloatArray,
    emotion_probabilities: FloatArray,
) -> FloatArray:
    """Combine visual recognition features and emotion probabilities.

    Parameters:
        image_embeddings: Learned CNN feature vectors for artwork images.
        emotion_probabilities: CNN softmax distributions over emotion labels.

    Returns:
        A unit-normalized representation matrix used for KNN distances.

    Raises:
        ValueError: If arrays have unequal row counts or contain invalid values.
    """
    if image_embeddings.ndim != 2 or emotion_probabilities.ndim != 2:
        raise ValueError("CNN embeddings and emotion probabilities must be matrices")
    if image_embeddings.shape[0] != emotion_probabilities.shape[0]:
        raise ValueError("CNN embeddings and emotion probabilities must have equal row counts")
    if not np.isfinite(image_embeddings).all() or not np.isfinite(emotion_probabilities).all():
        raise ValueError("CNN outputs contain a non-finite value")
    image_vectors = _normalize_rows(image_embeddings)
    emotion_vectors = _normalize_rows(emotion_probabilities)
    joined = np.concatenate((
        param.EMBEDDING_WEIGHT * image_vectors,
        param.RECOGNITION_WEIGHT * emotion_vectors,
    ), axis=1)
    return _normalize_rows(joined)


def _knn_graph(
    representations: FloatArray,
    neighbor_count: int,
) -> tuple[sparse.csr_matrix, NearestNeighbors]:
    """Construct a symmetric, similarity-weighted KNN graph.

    Parameters:
        representations: Joint visual and emotion representation matrix.
        neighbor_count: Number of neighbors to connect for each artwork.

    Returns:
        Sparse affinity matrix and fitted nearest-neighbor search model.

    Raises:
        ValueError: If fewer than four points are provided or neighbor_count
            is outside the valid range.
    """
    point_count = representations.shape[0]
    if point_count < 4:
        raise ValueError(f"At least four datapoints are required for a 3D map; found {point_count}")
    if neighbor_count < 1 or neighbor_count >= point_count:
        raise ValueError(
            f"neighbor_count must be in [1, {point_count - 1}]; received {neighbor_count}"
        )
    search = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="cosine")
    search.fit(representations)
    distances, indices = search.kneighbors(representations)
    row_indices = np.repeat(np.arange(point_count), neighbor_count)
    neighbor_indices = indices[:, 1:].reshape(-1)
    neighbor_distances = distances[:, 1:].reshape(-1)
    positive_distances = neighbor_distances[neighbor_distances > 0.0]
    scale = float(np.median(positive_distances)) if positive_distances.size else 1.0
    edge_weights = np.exp(-np.square(neighbor_distances / max(scale, 1e-8))).astype(np.float32)
    directed = sparse.csr_matrix(
        (edge_weights, (row_indices, neighbor_indices)),
        shape=(point_count, point_count),
        dtype=np.float32,
    )
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    if affinity.nnz == 0:
        raise ValueError("KNN graph contains no edges; check the recognition vectors")
    return affinity, search


def _embed_knn_space(
    affinity: sparse.csr_matrix,
    cluster_count: int,
    seed: int,
) -> tuple[FloatArray, IntArray]:
    """Project the KNN graph into 3D and assign graph-based emotion groups.

    Parameters:
        affinity: Sparse symmetric KNN similarity graph.
        cluster_count: Requested number of spectral emotion groups.
        seed: Seed used by stochastic eigensolver and clustering steps.

    Returns:
        Three-dimensional point coordinates and integer KNN cluster IDs.

    Raises:
        ValueError: If the graph or cluster count is invalid.
    """
    point_count = affinity.shape[0]
    if affinity.shape[1] != point_count or point_count < 4:
        raise ValueError(f"Expected a square graph with at least four nodes; got {affinity.shape}")
    if cluster_count < 2 or cluster_count >= point_count:
        raise ValueError(f"cluster_count must be in [2, {point_count - 1}]")
    embedding = SpectralEmbedding(
        n_components=3,
        affinity="precomputed",
        random_state=seed,
    ).fit_transform(affinity)
    clustering = SpectralClustering(
        n_clusters=cluster_count,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    )
    cluster_ids = clustering.fit_predict(affinity)
    return embedding.astype(np.float32), cluster_ids.astype(np.int64)


def _project_query(
    query_embedding: FloatArray,
    query_probabilities: FloatArray,
    reference_representation: FloatArray,
    reference_coordinates: FloatArray,
    reference_clusters: IntArray,
    search: NearestNeighbors,
    neighbor_count: int,
) -> tuple[FloatArray, int, IntArray]:
    """Place a new artwork by weighted interpolation from its K nearest works.

    Parameters:
        query_embedding: CNN embedding for the new artwork, shape ``(1, d)``.
        query_probabilities: CNN emotion distribution, shape ``(1, c)``.
        reference_representation: Joint representation matrix for known works.
        reference_coordinates: 3D KNN-map coordinates for known works.
        reference_clusters: KNN cluster IDs associated with known works.
        search: Fitted KNN search model for known works.
        neighbor_count: Number of known datapoints used to position the query.

    Returns:
        Query's interpolated 3D coordinate, majority neighbor cluster, and
        indices of the contributing known artworks.

    Raises:
        ValueError: If query shapes are invalid or neighbor_count is invalid.
    """
    if query_embedding.shape[0] != 1 or query_probabilities.shape[0] != 1:
        raise ValueError("A new-artwork projection accepts exactly one query image")
    query_representation = _joint_representation(query_embedding, query_probabilities)
    available = reference_representation.shape[0]
    if neighbor_count < 1 or neighbor_count > available:
        raise ValueError(f"Query neighbor_count must be in [1, {available}]")
    distances, indices = search.kneighbors(query_representation, n_neighbors=neighbor_count)
    selected_distances = distances[0]
    selected_indices = indices[0]
    exact = selected_distances <= 1e-8
    if exact.any():
        weights = exact.astype(np.float32)
    else:
        inverse_distances = 1.0 / np.maximum(selected_distances, 1e-8)
        weights = (inverse_distances / inverse_distances.sum()).astype(np.float32)
    coordinate = np.average(reference_coordinates[selected_indices], axis=0, weights=weights)
    neighbor_labels = reference_clusters[selected_indices]
    cluster_ids, cluster_counts = np.unique(neighbor_labels, return_counts=True)
    query_cluster = int(cluster_ids[np.argmax(cluster_counts)])
    return coordinate.astype(np.float32), query_cluster, selected_indices.astype(np.int64)


def _render_space(
    rows: pd.DataFrame,
    coordinates: FloatArray,
    clusters: IntArray,
    predicted_emotions: list[str],
    probabilities: FloatArray,
    output_path: Path,
    query_point: FloatArray | None,
    query_cluster: int | None,
    query_label: str | None,
) -> None:
    """Write a standalone mouse-rotatable Plotly 3D emotion map.

    Parameters:
        rows: Known artwork metadata aligned with coordinates.
        coordinates: KNN-manifold coordinates for the known artworks.
        clusters: Spectral KNN cluster IDs aligned with rows.
        predicted_emotions: Top CNN emotion prediction for each known artwork.
        probabilities: CNN class distributions aligned with rows.
        output_path: Destination HTML path.
        query_point: Optional interpolated coordinate for one new artwork.
        query_cluster: KNN group assigned to an optional query artwork.
        query_label: Display label for an optional query artwork.

    Returns:
        None. Writes an interactive HTML file to output_path.

    Raises:
        ValueError: If known-point arrays do not align or query metadata is
            incomplete.
        OSError: If the HTML output cannot be written.
    """
    count = len(rows)
    if coordinates.shape != (count, 3) or clusters.shape[0] != count:
        raise ValueError("Each known artwork must have one coordinate and cluster")
    if len(predicted_emotions) != count or probabilities.shape[0] != count:
        raise ValueError("Each known artwork must have one prediction probability vector")
    hover_text = [
        f"<b>{Path(str(row.image_path)).name}</b><br>"
        f"Dataset emotion: {row.dominant_emotion}<br>"
        f"CNN recognition: {predicted_emotions[index]}<br>"
        f"Style: {row.style}<br>Split: {row.split}<br>"
        f"KNN group: {int(clusters[index])}<br>"
        f"Recognition confidence: {float(probabilities[index].max()):.3f}"
        for index, row in enumerate(rows.itertuples(index=False))
    ]
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        z=coordinates[:, 2],
        mode="markers",
        text=hover_text,
        hoverinfo="text",
        marker={
            "size": 4,
            "opacity": 0.78,
            "color": clusters,
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {"title": "KNN group"},
        },
        name="Known artworks",
    ))
    if query_point is not None:
        if query_cluster is None or query_label is None:
            raise ValueError("A query coordinate requires its cluster and display label")
        figure.add_trace(go.Scatter3d(
            x=[float(query_point[0])],
            y=[float(query_point[1])],
            z=[float(query_point[2])],
            mode="markers+text",
            text=[query_label],
            textposition="top center",
            hovertemplate=f"{query_label}<br>KNN group: {query_cluster}<extra></extra>",
            marker={"size": 9, "color": "#111827", "symbol": "diamond"},
            name="New artwork, placed by KNN",
        ))
    figure.update_layout(
        title="EmoArt emotion space from CNN recognition and KNN relationships",
        template="plotly_white",
        scene={
            "xaxis_title": "KNN manifold axis 1",
            "yaxis_title": "KNN manifold axis 2",
            "zaxis_title": "KNN manifold axis 3",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def _parse_arguments() -> VisualizationArguments:
    """Read visualization options from the command line.

    Parameters:
        None.

    Returns:
        Namespace containing image query, neighbor count, cluster count,
        output path, and random seed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--neighbors", type=int, default=param.K_NEIGHBORS)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--output", type=Path, default=param.VISUALIZATION_DIR / "emotion_space.html")
    parser.add_argument("--seed", type=int, default=param.RANDOM_SEED)
    parsed = parser.parse_args()
    return VisualizationArguments(
        image=parsed.image,
        neighbors=int(parsed.neighbors),
        clusters=int(parsed.clusters),
        output=Path(parsed.output),
        seed=int(parsed.seed),
    )


def main() -> None:
    """Generate the 3D KNN map and optionally locate one new artwork.

    Parameters:
        None.

    Returns:
        None. Writes an interactive HTML map and coordinate CSV.

    Raises:
        FileNotFoundError: If annotations, checkpoints, labels, or query image
            files are missing.
        ValueError: If split membership, model dimensions, or KNN parameters
            are invalid.
    """
    arguments = _parse_arguments()
    seed_everything(arguments.seed)
    checkpoint = param.OUTPUT_DIR / "best_model.pt"
    labels_file = param.OUTPUT_DIR / "labels.json"
    if not checkpoint.is_file() or not labels_file.is_file() or not param.SPLIT_FILE.is_file():
        raise FileNotFoundError("Train the CNN first; checkpoint, label list, or saved data split is missing")
    if arguments.image is not None and not arguments.image.is_file():
        raise FileNotFoundError(f"Query artwork image does not exist: {arguments.image}")
    class_names = json.loads(labels_file.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError(f"Invalid emotion vocabulary in {labels_file}")
    emotion_to_index = {str(name): index for index, name in enumerate(class_names)}
    splits = pd.read_csv(param.SPLIT_FILE)
    known_rows = splits.loc[splits["split"].isin(("train", "validation"))].reset_index(drop=True)
    if known_rows.empty:
        raise ValueError("Saved train and validation splits have no known artworks")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtEmotionCNN(len(class_names), embedding_dim=param.EMBEDDING_DIM, dropout=param.DROPOUT)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    embeddings, probabilities = _extract_recognition(
        known_rows, model, emotion_to_index, device, param.BATCH_SIZE
    )
    representations = _joint_representation(embeddings, probabilities)
    affinity, search = _knn_graph(representations, arguments.neighbors)
    cluster_count = min(arguments.clusters, len(known_rows) - 1)
    coordinates, clusters = _embed_knn_space(affinity, cluster_count, arguments.seed)
    top_indices = probabilities.argmax(axis=1)
    predicted_emotions = [class_names[int(index)] for index in top_indices]
    query_point: FloatArray | None = None
    query_cluster: int | None = None
    query_label: str | None = None
    if arguments.image is not None:
        query_frame = pd.DataFrame([{
            "image_path": str(arguments.image.resolve()),
            "dominant_emotion": class_names[0],
            "valence": 0,
            "arousal": 0,
            "request_id": "query",
            "style": "new artwork",
            "split": "query",
        }])
        query_embedding, query_probabilities = _extract_recognition(
            query_frame, model, emotion_to_index, device, 1
        )
        query_point, query_cluster, _ = _project_query(
            query_embedding,
            query_probabilities,
            representations,
            coordinates,
            clusters,
            search,
            min(arguments.neighbors, len(known_rows)),
        )
        query_label = (
            f"New artwork: {class_names[int(query_probabilities[0].argmax())]} "
            f"({float(query_probabilities[0].max()):.2f})"
        )
    _render_space(
        known_rows,
        coordinates,
        clusters,
        predicted_emotions,
        probabilities,
        arguments.output,
        query_point,
        query_cluster,
        query_label,
    )
    output_frame = known_rows.copy()
    output_frame["x"] = coordinates[:, 0]
    output_frame["y"] = coordinates[:, 1]
    output_frame["z"] = coordinates[:, 2]
    output_frame["knn_group"] = clusters
    output_frame["predicted_emotion"] = predicted_emotions
    output_frame["emotion_confidence"] = probabilities.max(axis=1)
    csv_path = arguments.output.with_suffix(".csv")
    output_frame.to_csv(csv_path, index=False)
    print(f"Interactive renderer: {arguments.output}")
    print(f"Known-artwork coordinates: {csv_path}")
    if arguments.image is not None:
        print(f"Query emotion: {query_label}; assigned KNN group {query_cluster}")


if __name__ == "__main__":
    main()
