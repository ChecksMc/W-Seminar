"""CNN that recognizes dataset emotion labels and exposes visual embeddings."""

from torch import Tensor
from torch import nn


class ArtEmotionCNN(nn.Module):
    """Predict a dominant emotion and return an artwork feature embedding.

    Parameters:
        emotion_count: Number of distinct dominant-emotion target categories.
        embedding_dim: Number of values in the learned artwork representation.
        dropout: Dropout probability applied before the emotion-classification
            layer.

    Returns:
        A PyTorch module whose forward result is emotion logits and an
        L2-normalized feature embedding.

    Raises:
        ValueError: If the class count, embedding size, or dropout rate is
            outside its valid range.
    """

    def __init__(
        self,
        emotion_count: int,
        embedding_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if emotion_count < 2:
            raise ValueError(f"At least two emotion classes are required; received {emotion_count}")
        if embedding_dim <= 0:
            raise ValueError(f"Embedding size must be positive; received {embedding_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout must be in [0, 1); received {dropout}")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, embedding_dim),
            nn.ReLU(),
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embedding_dim, emotion_count)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Compute emotion logits and a normalized learned image embedding.

        Parameters:
            images: Batch tensor shaped ``(batch, 3, height, width)`` with
                normalized RGB pixel values.

        Returns:
            A pair containing class logits shaped ``(batch, emotion_count)``
            and normalized embeddings shaped ``(batch, embedding_dim)``.

        Raises:
            ValueError: If the input is not a four-dimensional RGB batch.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected an RGB image batch with shape (N, 3, H, W); got {tuple(images.shape)}")
        features = self.features(images)
        embedding = self.embedding_layer(features)
        logits = self.classifier(self.dropout_layer(embedding))
        normalized_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return logits, normalized_embedding
