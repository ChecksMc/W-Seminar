"""Recognize dominant emotion labels in one artwork with the trained CNN."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from . import param
from .model import ArtEmotionCNN


@dataclass(frozen=True)
class PredictionArguments:
    """Command-line values needed to recognize one artwork.

    Parameters:
        image: Artwork file to classify.
        checkpoint: Trained CNN parameter file.
        top_k: Number of highest-probability emotions to print.

    Returns:
        An immutable prediction configuration object.
    """

    image: Path
    checkpoint: Path
    top_k: int


def _parse_arguments() -> PredictionArguments:
    """Parse the artwork path and checkpoint options.

    Parameters:
        None.

    Returns:
        Namespace containing image, checkpoint, and top-k settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=param.OUTPUT_DIR / "best_model.pt")
    parser.add_argument("--top-k", type=int, default=5)
    parsed = parser.parse_args()
    return PredictionArguments(
        image=Path(parsed.image),
        checkpoint=Path(parsed.checkpoint),
        top_k=int(parsed.top_k),
    )


def _load_image(path: Path, device: torch.device) -> Tensor:
    """Load and transform one artwork image for the CNN.

    Parameters:
        path: Existing image path to classify.
        device: Torch device to which the transformed image batch is moved.

    Returns:
        Normalized RGB image tensor with shape ``(1, 3, image_size, image_size)``.

    Raises:
        FileNotFoundError: If path does not refer to a file.
        OSError: If the file is not a readable image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Artwork image does not exist: {path}")
    transform = transforms.Compose([
        transforms.Resize((param.IMAGE_SIZE, param.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:
        with Image.open(path) as source:
            return transform(source.convert("RGB")).unsqueeze(0).to(device)
    except OSError as error:
        raise OSError(f"Could not read artwork image: {path}") from error


def main() -> None:
    """Load the trained CNN and print the highest emotion-label probabilities.

    Parameters:
        None.

    Returns:
        None. Prints the ranked emotion labels and probabilities.

    Raises:
        FileNotFoundError: If the image, checkpoint, or class-label file is
            missing.
        ValueError: If top-k is invalid or stored class labels are malformed.
    """
    arguments = _parse_arguments()
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(f"CNN checkpoint does not exist: {arguments.checkpoint}")
    labels_path = param.OUTPUT_DIR / "labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Emotion label file does not exist: {labels_path}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"Invalid emotion labels in {labels_path}")
    if arguments.top_k <= 0 or arguments.top_k > len(labels):
        raise ValueError(f"--top-k must be between 1 and {len(labels)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtEmotionCNN(len(labels), embedding_dim=param.EMBEDDING_DIM)
    model.load_state_dict(torch.load(arguments.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(_load_image(arguments.image, device))
        probabilities = torch.softmax(logits, dim=1)[0]
    top_probabilities, top_indices = torch.topk(probabilities, k=arguments.top_k)
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist(), strict=True):
        print(f"{labels[index]}: {probability:.3f}")
    print("These are EmoArt label predictions, not measurements of a viewer's felt emotion.")


if __name__ == "__main__":
    main()
