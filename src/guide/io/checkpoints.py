"""Load native or losslessly compressed PyTorch model checkpoints."""

from __future__ import annotations

from collections import OrderedDict
import zlib

import numpy as np
import torch


def _decode_bytes(payload: torch.Tensor, dtype: np.dtype, count: int) -> np.ndarray:
    """Undo zlib compression and byte shuffling without converting values."""
    if payload.dtype != torch.uint8 or payload.ndim != 1:
        raise ValueError("Compressed tensor data must be a one-dimensional byte tensor.")
    raw = zlib.decompress(payload.numpy().tobytes())
    if len(raw) != count * dtype.itemsize:
        raise ValueError("Compressed tensor size does not match its shape and dtype.")
    return (
        np.frombuffer(raw, dtype=np.uint8)
        .reshape(dtype.itemsize, count).T.copy().reshape(-1).view(dtype)
    )


def _decode_tensor(entry: dict) -> torch.Tensor:
    shape = tuple(entry["shape"])
    dtype = np.dtype(entry["dtype"])
    if any(not isinstance(n, int) or n < 0 for n in shape) or dtype.kind not in "fiub":
        raise ValueError("Invalid compressed tensor shape or dtype.")
    if entry.get("layout", "dense") == "dense":
        array = _decode_bytes(entry["data"], dtype, int(np.prod(shape))).reshape(shape)
    elif entry["layout"] == "upper_xor":
        if len(shape) != 2 or shape[0] != shape[1] or dtype != np.dtype("float64"):
            raise ValueError("Upper/XOR storage requires a square FP64 matrix.")
        n = shape[0]
        upper = _decode_bytes(entry["upper"], np.dtype("uint64"), n * (n + 1) // 2)
        xor = _decode_bytes(entry["xor"], np.dtype("uint64"), n * (n - 1) // 2)
        bits = np.empty(shape, dtype=np.uint64)
        upper_offset = xor_offset = 0
        for row in range(n):
            width = n - row
            values = upper[upper_offset:upper_offset + width]
            bits[row, row:] = values
            bits[row + 1:, row] = values[1:] ^ xor[xor_offset:xor_offset + width - 1]
            upper_offset += width
            xor_offset += width - 1
        array = bits.view(dtype)
    else:
        raise ValueError(f"Unsupported tensor layout: {entry['layout']}")
    return torch.from_numpy(array)


def load_model_checkpoint(path):
    """Restore exact tensor bytes on CPU; also accept native state dictionaries."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "guide-tensor-zlib-v1":
        return checkpoint
    state = OrderedDict(
        (name, _decode_tensor(entry))
        for name, entry in checkpoint["packed_state_dict"].items()
    )
    if "state_dict_metadata" in checkpoint:
        state._metadata = checkpoint["state_dict_metadata"]
    restored = {"model_state_dict": state}
    for key in ("model_config", "network_config"):
        if key in checkpoint:
            restored[key] = checkpoint[key]
    return restored
