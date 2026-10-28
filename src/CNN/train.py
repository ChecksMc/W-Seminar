"""Train, tune, and test the emotion-recognition CNN using fixed data splits."""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from . import param
from .data import EmoArtDataset, save_or_load_splits, seed_everything
from .model import ArtEmotionCNN


@dataclass(frozen=True)
class TrialConfig:
    """Hyperparameter values for one model training run.

    Parameters:
        learning_rate: AdamW learning rate.
        weight_decay: AdamW L2-style decoupled weight decay.
        dropout: Drop probability before the emotion classifier.
        epochs: Maximum full passes over the training split.
        batch_size: Number of images in one optimization step.

    Returns:
        An immutable configuration value object.
    """

    learning_rate: float
    weight_decay: float
    dropout: float
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class TrainArguments:
    """Command-line values controlling one training or tuning run.

    Parameters:
        epochs: Maximum epochs trained for each configuration.
        batch_size: Number of images per training batch.
        tune: Whether to compare the predefined hyperparameter grid.
        seed: Random seed for data loading and model initialization.

    Returns:
        An immutable command-line configuration object.
    """

    epochs: int
    batch_size: int
    tune: bool
    seed: int


def _make_loader(
    frame: pd.DataFrame,
    emotion_to_index: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a PyTorch image loader for one dataframe partition.

    Parameters:
        frame: Annotation rows belonging to one split.
        emotion_to_index: Stable emotion string to class-index mapping.
        batch_size: Number of examples yielded in each batch.
        shuffle: Whether examples are reshuffled each iteration.

    Returns:
        A data loader that yields image batches and integer class targets.

    Raises:
        ValueError: If the partition is empty or the batch size is not positive.
    """
    if frame.empty:
        raise ValueError("Cannot create a data loader for an empty split")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive; received {batch_size}")
    dataset = EmoArtDataset(frame, emotion_to_index)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _class_weights(frame: pd.DataFrame, class_names: list[str], device: torch.device) -> Tensor:
    """Create normalized inverse-frequency weights for emotion categories.

    Parameters:
        frame: Training partition used to estimate class frequencies.
        class_names: Complete ordered emotion-label vocabulary.
        device: Device receiving the resulting tensor.

    Returns:
        A float tensor with one positive weight for each emotion class.

    Raises:
        ValueError: If a class is absent from training or training rows are empty.
    """
    if frame.empty:
        raise ValueError("Cannot calculate class weights without training examples")
    counts = frame["dominant_emotion"].value_counts()
    missing = [name for name in class_names if int(counts.get(name, 0)) == 0]
    if missing:
        raise ValueError(f"The training split has no examples for emotion labels: {missing}")
    frequencies = torch.tensor([float(counts[name]) for name in class_names], dtype=torch.float32)
    weights = frequencies.sum() / (len(class_names) * frequencies)
    return (weights / weights.mean()).to(device)


def _evaluate(
    model: ArtEmotionCNN,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    class_count: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Calculate loss and class-balanced emotion-recognition metrics.

    Parameters:
        model: CNN being evaluated.
        loader: Validation or test data loader.
        loss_function: Weighted cross-entropy loss module.
        class_count: Number of distinct emotion labels.
        device: Torch device used for inference.

    Returns:
        Tuple of mean loss and metrics containing accuracy, balanced accuracy,
        and macro F1.

    Raises:
        ValueError: If the loader yields no examples or class_count is invalid.
    """
    if class_count < 2:
        raise ValueError(f"At least two classes are required for evaluation; received {class_count}")
    model.eval()
    accumulated_loss = 0.0
    target_values: list[int] = []
    prediction_values: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits, _ = model(images)
            accumulated_loss += float(loss_function(logits, targets).item()) * len(targets)
            prediction_values.extend(logits.argmax(dim=1).cpu().tolist())
            target_values.extend(targets.cpu().tolist())
    if not target_values:
        raise ValueError("Evaluation split did not yield any image examples")
    labels = np.arange(class_count)
    metrics = {
        "accuracy": float(accuracy_score(target_values, prediction_values)),
        "balanced_accuracy": float(balanced_accuracy_score(target_values, prediction_values)),
        "macro_f1": float(f1_score(
            target_values,
            prediction_values,
            labels=labels,
            average="macro",
            zero_division=0,
        )),
    }
    return accumulated_loss / len(target_values), metrics


def _fit_trial(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    emotion_to_index: dict[str, int],
    class_names: list[str],
    config: TrialConfig,
    device: torch.device,
) -> tuple[ArtEmotionCNN, float, dict[str, float]]:
    """Fit one CNN configuration and select its best validation checkpoint.

    Parameters:
        train_rows: Rows used to update neural-network weights.
        validation_rows: Unseen rows used to select the epoch checkpoint.
        emotion_to_index: Stable mapping from emotion label to model output.
        class_names: Ordered class vocabulary.
        config: Learning-rate, regularization, dropout, batch, and epoch setup.
        device: Torch device used for training and validation.

    Returns:
        Best model, its validation macro F1, and its last validation metrics.

    Raises:
        ValueError: If configuration values are invalid or a split is empty.
    """
    if config.epochs <= 0 or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError(f"Invalid trial configuration: {config}")
    if config.batch_size <= 0 or not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"Invalid batch size or dropout in trial configuration: {config}")
    train_loader = _make_loader(train_rows, emotion_to_index, config.batch_size, True)
    validation_loader = _make_loader(validation_rows, emotion_to_index, config.batch_size, False)
    model = ArtEmotionCNN(
        emotion_count=len(class_names),
        embedding_dim=param.EMBEDDING_DIM,
        dropout=config.dropout,
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=_class_weights(train_rows, class_names, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, float] = {}
    for epoch in range(config.epochs):
        model.train()
        accumulated_loss = 0.0
        sample_count = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.item()) * len(targets)
            sample_count += len(targets)
        validation_loss, validation_metrics = _evaluate(
            model, validation_loader, loss_function, len(class_names), device
        )
        print(
            f"epoch={epoch + 1:02}/{config.epochs} "
            f"train_loss={accumulated_loss / sample_count:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_metrics = validation_metrics
    if best_state is None:
        raise RuntimeError("Training completed without producing a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_f1, best_metrics


def _trial_configs(epochs: int, batch_size: int) -> list[TrialConfig]:
    """Return a small deterministic grid of training hyperparameters.

    Parameters:
        epochs: Maximum epochs per candidate configuration.
        batch_size: Images per training batch for each candidate.

    Returns:
        Four configurations covering learning rate and dropout combinations.

    Raises:
        ValueError: If epochs or batch_size is not positive.
    """
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Epoch count and batch size must both be positive")
    return [
        TrialConfig(learning_rate, 1e-4, dropout, epochs, batch_size)
        for learning_rate, dropout in product((1e-3, 3e-4), (0.2, 0.4))
    ]


def _parse_arguments() -> TrainArguments:
    """Parse options controlling baseline training and hyperparameter tuning.

    Parameters:
        None.

    Returns:
        Namespace containing the requested epoch count, batch size, and tuning
        flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=param.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=param.BATCH_SIZE)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--seed", type=int, default=param.RANDOM_SEED)
    parsed = parser.parse_args()
    return TrainArguments(
        epochs=int(parsed.epochs),
        batch_size=int(parsed.batch_size),
        tune=bool(parsed.tune),
        seed=int(parsed.seed),
    )


def main() -> None:
    """Train or tune the CNN, select by validation macro F1, then test once.

    Parameters:
        None.

    Returns:
        None. Checkpoints and JSON metrics are written under ``runs/cnn``.

    Raises:
        FileNotFoundError: If EmoArt annotations or images are not installed.
        ValueError: If the saved data splits, labels, or training parameters
            are invalid.
        RuntimeError: If no checkpoint can be selected during training.
    """
    arguments = _parse_arguments()
    seed_everything(arguments.seed)
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive integers")
    param.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = save_or_load_splits()
    train_rows = split_rows.loc[split_rows["split"] == "train"].copy()
    validation_rows = split_rows.loc[split_rows["split"] == "validation"].copy()
    test_rows = split_rows.loc[split_rows["split"] == "test"].copy()
    if train_rows.empty or validation_rows.empty or test_rows.empty:
        raise ValueError("Train, validation, and test partitions must all contain records")
    class_names = sorted(str(label) for label in split_rows["dominant_emotion"].unique())
    if len(class_names) < 2:
        raise ValueError("At least two distinct dominant-emotion labels are required")
    emotion_to_index = {name: index for index, name in enumerate(class_names)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = (
        _trial_configs(arguments.epochs, arguments.batch_size)
        if arguments.tune
        else [TrialConfig(
            param.LEARNING_RATE,
            param.WEIGHT_DECAY,
            param.DROPOUT,
            arguments.epochs,
            arguments.batch_size,
        )]
    )
    results: list[dict[str, Any]] = []
    best_model: ArtEmotionCNN | None = None
    best_config: TrialConfig | None = None
    best_validation_f1 = -1.0
    for trial_number, config in enumerate(configs, start=1):
        print(f"trial={trial_number}/{len(configs)} configuration={asdict(config)}")
        seed_everything(arguments.seed)
        model, validation_f1, validation_metrics = _fit_trial(
            train_rows,
            validation_rows,
            emotion_to_index,
            class_names,
            config,
            device,
        )
        results.append({
            "configuration": asdict(config),
            "validation_macro_f1": validation_f1,
            "validation_metrics": validation_metrics,
        })
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_model = model
            best_config = config
    if best_model is None or best_config is None:
        raise RuntimeError("No model was selected from the training configurations")
    best_model.to(device)
    class_counts = train_rows["dominant_emotion"].value_counts().to_dict()
    missing_classes = [name for name in class_names if int(class_counts.get(name, 0)) == 0]
    if missing_classes:
        raise ValueError(f"No training examples exist for labels: {missing_classes}")
    final_loss_function = nn.CrossEntropyLoss(
        weight=_class_weights(train_rows, class_names, device)
    )
    test_loss, test_metrics = _evaluate(
        best_model,
        _make_loader(test_rows, emotion_to_index, best_config.batch_size, False),
        final_loss_function,
        len(class_names),
        device,
    )
    torch.save(best_model.state_dict(), param.OUTPUT_DIR / "best_model.pt")
    (param.OUTPUT_DIR / "labels.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "selected_configuration": asdict(best_config),
        "validation_macro_f1": best_validation_f1,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "emotion_labels": class_names,
        "train_class_counts": class_counts,
        "split_sizes": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "tuning_trials": results,
        "target_caveat": "EmoArt emotion labels are not measurements of an individual viewer's felt emotion.",
    }
    (param.OUTPUT_DIR / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
