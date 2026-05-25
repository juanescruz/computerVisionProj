"""Bilateral filter implementation."""
import numpy as np


def bilateral_filter_gray(img: np.ndarray, sigma_spatial: float = 5,
                           sigma_range: float = 50) -> np.ndarray:
    """Bilateral filter for grayscale images."""
    img = img.astype(np.float32)
    h, w = img.shape

    kernel_size = int(6 * sigma_spatial + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2

    kernel_spatial = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - radius
            y = j - radius
            kernel_spatial[i, j] = np.exp(
                -(x * x + y * y) / (2 * sigma_spatial * sigma_spatial)
            )

    padded = np.pad(img, radius, mode='edge')
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            center_val = img[i, j]

            kernel_range = np.exp(
                -((window - center_val) ** 2) / (2 * sigma_range * sigma_range)
            )

            weights = kernel_spatial * kernel_range
            wsum = np.sum(weights)

            if wsum > 0:
                output[i, j] = np.sum(window * weights) / wsum
            else:
                output[i, j] = center_val

    return np.clip(output, 0, 255).astype(np.uint8)


def bilateral_filter_color(img: np.ndarray, sigma_spatial: float = 5,
                            sigma_range: float = 50) -> np.ndarray:
    """Bilateral filter for RGB images."""
    h, w, c = img.shape
    img = img.astype(np.float32)

    kernel_size = int(6 * sigma_spatial + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2

    kernel_spatial = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - radius
            y = j - radius
            kernel_spatial[i, j] = np.exp(
                -(x * x + y * y) / (2 * sigma_spatial * sigma_spatial)
            )

    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)),
                     mode='edge')
    output = np.zeros((h, w, c), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + kernel_size, j:j + kernel_size, :]
            center_val = img[i, j]

            diff = window - center_val
            dist_color = np.sqrt(np.sum(diff ** 2, axis=2))
            kernel_range = np.exp(
                -(dist_color ** 2) / (2 * sigma_range * sigma_range)
            )

            for chan in range(c):
                weights = kernel_spatial * kernel_range
                wsum = np.sum(weights)
                if wsum > 0:
                    output[i, j, chan] = (
                        np.sum(window[:, :, chan] * weights) / wsum
                    )
                else:
                    output[i, j, chan] = center_val[chan]

    return np.clip(output, 0, 255).astype(np.uint8)


def bilateral_filter(img: np.ndarray, sigma_spatial: float = 5,
                     sigma_range: float = 50) -> np.ndarray:
    """Wrapper: detects grayscale vs color."""
    if len(img.shape) == 2:
        return bilateral_filter_gray(img, sigma_spatial, sigma_range)
    return bilateral_filter_color(img, sigma_spatial, sigma_range)
