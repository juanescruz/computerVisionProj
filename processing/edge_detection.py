"""Gradient edge detectors using NumPy."""
import numpy as np


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D convolution manually."""
    h, w = img.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="edge"
    )
    output = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return output


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude."""
    gradient = np.sqrt(gx**2 + gy**2)
    gradient = np.clip(gradient, 0, 255)
    return gradient.astype(np.uint8)


def prewitt_operator(img: np.ndarray) -> np.ndarray:
    """Apply Prewitt edge detector."""
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    gx = convolve2d(img, kernel_x)
    gy = convolve2d(img, kernel_y)

    return gradient_magnitude(gx, gy)


def sobel_operator(img: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detector."""
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    gx = convolve2d(img, kernel_x)
    gy = convolve2d(img, kernel_y)

    return gradient_magnitude(gx, gy)