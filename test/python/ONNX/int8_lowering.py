"""Small NumPy helpers for compiler-side INT8 lowering tests."""

import numpy as np


def conv_to_matrix(
    input_value: np.ndarray,
    kernel: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    """Return NCHW convolution patches as ``[K, spatial]``."""
    if input_value.ndim != 4 or input_value.shape[0] != 1:
        raise ValueError("input must have shape [1,C,H,W]")
    pad_top, pad_left, pad_bottom, pad_right = pads
    padded = np.pad(
        input_value,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded[0], kernel, axis=(1, 2)
    )
    sampled = windows[:, :: strides[0], :: strides[1], :, :]
    return (
        sampled.transpose(1, 2, 0, 3, 4)
        .reshape(sampled.shape[1] * sampled.shape[2], -1)
        .T
    )


def fold_bias(
    weight: np.ndarray,
    input_value: np.ndarray,
    bias: np.ndarray,
    exact_k: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a constant-one input lane carrying an INT8 bias."""
    bias_column = bias.astype(np.int8, copy=False).reshape(-1, 1)
    one_row = np.ones((1, input_value.shape[1]), dtype=np.int8)
    if exact_k:
        return (
            np.concatenate(
                (weight[:, :-1], bias_column, weight[:, -1:]), axis=1
            ),
            np.concatenate(
                (input_value[:-1], one_row, input_value[-1:]), axis=0
            ),
        )
    return (
        np.concatenate((weight, bias_column), axis=1),
        np.concatenate((input_value, one_row), axis=0),
    )


def wrap_i32(values: np.ndarray) -> np.ndarray:
    """Apply ONNX's two's-complement INT32 result conversion."""
    return ((values + 2**31) % 2**32 - 2**31).astype(np.int32)
