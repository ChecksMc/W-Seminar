"""Resolve an explicit CPU, CUDA, or automatic PyTorch device choice."""

from typing import Literal

import torch

DeviceSelection = Literal["auto", "cpu", "nvidia", "cuda"]


def resolve_device(selection: DeviceSelection) -> torch.device:
    """Select the device used for CNN training or inference.

    Parameters:
        selection: ``auto`` selects CUDA when available and CPU otherwise;
            ``cpu`` forces CPU; ``nvidia`` and ``cuda`` require an available
            CUDA device.

    Returns:
        The selected PyTorch device.

    Raises:
        ValueError: If selection is not auto, cpu, nvidia, or cuda.
        RuntimeError: If NVIDIA/CUDA was requested but PyTorch cannot use it.
    """
    if selection == "cpu":
        return torch.device("cpu")
    if selection in ("nvidia", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible NVIDIA GPU and driver "
                "are available to this PyTorch installation"
            )
        return torch.device("cuda")
    if selection == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device selection: {selection}")
