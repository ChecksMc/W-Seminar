"""Read EmoArt labels, create fixed data splits, and load artwork tensors."""

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from . import param


def _records(payload: Any) -> list[dict[str, Any]]:
    """Return annotation records from a supported JSON root structure.

    Parameters:
        payload: Parsed JSON data expected to be a list of records or a mapping
            containing a list under a known record key.

    Returns:
        A list of annotation dictionaries.

    Raises:
        ValueError: If the JSON root does not contain annotation records.
    """
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "annotations", "items", "images"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        values = list(payload.values())
        if values and all(isinstance(item, dict) for item in values):
            return values
    raise ValueError("Annotation JSON must contain a non-empty list of object records")


def read_annotations(path: Path = param.ANNOTATION_FILE) -> pd.DataFrame:
    """Load image records with emotion, valence, and arousal labels.

    Parameters:
        path: Path to EmoArt's annotation JSON file.

    Returns:
        A dataframe with image path, dominant emotion, binary valence and
        arousal, request identifier, and artwork style columns.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ValueError: If the JSON is malformed, no valid records remain, or a
            record has an invalid schema.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as annotation_stream:
            records = _records(json.load(annotation_stream))
    except json.JSONDecodeError as error:
        raise ValueError(f"Annotation file is not valid JSON: {path}") from error

    rows: list[dict[str, Any]] = []
    invalid_records = 0
    for record in records:
        description = record.get("description")
        third_section = description.get("third_section") if isinstance(description, dict) else None
        third_section = third_section if isinstance(third_section, dict) else {}
        raw_path = record.get("image_path") or record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            invalid_records += 1
            continue
        normalized_path = raw_path.replace("\\", "/")
        image_candidates = (
            param.RAW_DIR / normalized_path,
            param.IMAGE_DIR / Path(normalized_path).name,
        )
        image_path = next((candidate for candidate in image_candidates if candidate.is_file()), None)
        emotion = third_section.get("dominant_emotion")
        valence = str(third_section.get("emotional_valence", "")).strip().lower()
        arousal = str(third_section.get("emotional_arousal_level", "")).strip().lower()
        if image_path is None or not isinstance(emotion, str) or not emotion.strip():
            invalid_records += 1
            continue
        if valence not in {"positive", "negative"} or arousal not in {"high", "low"}:
            invalid_records += 1
            continue
        rows.append({
            "image_path": str(image_path),
            "dominant_emotion": emotion.strip(),
            "valence": int(valence == "positive"),
            "arousal": int(arousal == "high"),
            "request_id": str(record.get("request_id", "")),
            "style": Path(normalized_path).parent.name,
        })
    frame = pd.DataFrame(rows).drop_duplicates("image_path").reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"No valid labeled image records found in {path}; check the image archive and annotation schema"
        )
    if len(frame) < 10:
        raise ValueError(f"At least 10 valid artwork records are required; found {len(frame)}")
    if invalid_records:
        print(f"Skipped {invalid_records} annotation records with missing or invalid fields")
    return frame


def make_splits(frame: pd.DataFrame, seed: int = param.RANDOM_SEED) -> pd.DataFrame:
    """Create a deterministic 70/15/15 split stratified by valence-arousal pair.

    Parameters:
        frame: Non-empty dataframe returned by ``read_annotations``.
        seed: Random seed controlling split membership.

    Returns:
        A shuffled dataframe with a ``split`` column containing train,
        validation, and test assignments.

    Raises:
        ValueError: If the dataset is too small or too imbalanced to stratify.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty annotation dataframe")
    if len(frame) < 10:
        raise ValueError(f"At least 10 records are required for a 70/15/15 split; found {len(frame)}")
    fraction_total = param.TRAIN_FRACTION + param.VALIDATION_FRACTION + param.TEST_FRACTION
    if abs(fraction_total - 1.0) > 1e-8:
        raise ValueError(f"Train, validation, and test fractions must sum to 1; got {fraction_total}")
    labels = frame["valence"].astype(str) + frame["arousal"].astype(str)
    try:
        train_rows, remainder = train_test_split(
            frame,
            train_size=param.TRAIN_FRACTION,
            random_state=seed,
            stratify=labels,
        )
        remainder_labels = remainder["valence"].astype(str) + remainder["arousal"].astype(str)
        validation_rows, test_rows = train_test_split(
            remainder,
            train_size=param.VALIDATION_FRACTION / (param.VALIDATION_FRACTION + param.TEST_FRACTION),
            random_state=seed,
            stratify=remainder_labels,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified 70/15/15 split; provide more examples "
            "for each valence-arousal combination"
        ) from error
    present_training_emotions = set(train_rows["dominant_emotion"])
    for emotion in sorted(set(frame["dominant_emotion"]) - present_training_emotions):
        validation_candidates = validation_rows.loc[validation_rows["dominant_emotion"] == emotion]
        test_candidates = test_rows.loc[test_rows["dominant_emotion"] == emotion]
        source_rows = validation_candidates if not validation_candidates.empty else test_candidates
        if source_rows.empty:
            raise ValueError(f"Emotion label {emotion!r} has no example available for training")
        selected = source_rows.sample(n=1, random_state=seed)
        train_rows = pd.concat((train_rows, selected))
        validation_rows = validation_rows.drop(index=selected.index)
        test_rows = test_rows.drop(index=selected.index)
    partitions: list[pd.DataFrame] = []
    for split_name, partition in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        labeled_partition = partition.copy()
        labeled_partition["split"] = split_name
        partitions.append(labeled_partition)
    return pd.concat(partitions, ignore_index=True).sample(frac=1, random_state=seed)


class EmoArtDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load transformed artwork tensors and their recognized-emotion targets.

    Parameters:
        frame: Annotation dataframe containing image paths and dominant emotion.
        emotion_to_index: Mapping assigning each emotion label a contiguous
            model target index.
        image_size: Pixel height and width used to resize each RGB image.

    Returns:
        A dataset whose items are an image tensor and an integer class tensor.

    Raises:
        ValueError: If the dataframe is empty, the image size is invalid, or an
            emotion is missing from the supplied mapping.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        emotion_to_index: dict[str, int],
        image_size: int = param.IMAGE_SIZE,
    ) -> None:
        if frame.empty:
            raise ValueError("Cannot construct an image dataset from zero records")
        if image_size <= 0:
            raise ValueError(f"Image size must be positive; received {image_size}")
        unknown_emotions = set(frame["dominant_emotion"]) - set(emotion_to_index)
        if unknown_emotions:
            raise ValueError(f"Emotion mapping is missing labels: {sorted(unknown_emotions)}")
        self.frame = frame.reset_index(drop=True)
        self.emotion_to_index = emotion_to_index
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        """Return the number of available artwork examples.

        Returns:
            Number of rows in the dataset.
        """
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load one RGB artwork and convert its emotion to a class index.

        Parameters:
            index: Zero-based position of the requested artwork.

        Returns:
            A normalized image tensor and a scalar integer emotion target.

        Raises:
            IndexError: If the index is outside the dataset bounds.
            OSError: If the image cannot be read.
        """
        if index < 0 or index >= len(self.frame):
            raise IndexError(f"Dataset index {index} is outside [0, {len(self.frame)})")
        row = self.frame.iloc[index]
        try:
            with Image.open(str(row["image_path"])) as source:
                image = self.transform(source.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise OSError(f"Could not read artwork image: {row['image_path']}") from error
        target = torch.tensor(self.emotion_to_index[str(row["dominant_emotion"])], dtype=torch.long)
        return image, target


def save_or_load_splits(force_new: bool = False) -> pd.DataFrame:
    """Load a saved split or create and persist a reproducible one.

    Parameters:
        force_new: Whether to replace a previously saved split.

    Returns:
        Dataframe containing data records and their split assignment.

    Raises:
        FileNotFoundError: If annotations have not been downloaded.
        ValueError: If annotations cannot form a valid split.
    """
    param.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if param.SPLIT_FILE.is_file() and not force_new:
        saved = pd.read_csv(param.SPLIT_FILE)
        required_columns = {"image_path", "dominant_emotion", "split"}
        if not required_columns.issubset(saved.columns):
            raise ValueError(f"Saved split file is missing columns: {sorted(required_columns - set(saved.columns))}")
        if saved.empty:
            raise ValueError(f"Saved split file contains no rows: {param.SPLIT_FILE}")
        return saved
    split = make_splits(read_annotations())
    split.to_csv(param.SPLIT_FILE, index=False)
    return split


def seed_everything(seed: int = param.RANDOM_SEED) -> None:
    """Seed Python and PyTorch random generators for repeatable runs.

    Parameters:
        seed: Non-negative integer seed shared by the random generators.

    Returns:
        None.

    Raises:
        ValueError: If the supplied seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative; received {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
