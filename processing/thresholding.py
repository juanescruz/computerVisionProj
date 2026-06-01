"""Automatic thresholding methods: iterative, Otsu, and Otsu per RGB band."""
import numpy as np


def iterative_threshold(img: np.ndarray, tolerance: float = 1.0) -> tuple:
    """Iterative optimal threshold.
    Returns: (threshold, binary_image)
    """
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    img_flat = img.astype(np.float32).ravel()
    T = float(np.mean(img_flat))

    while True:
        G1 = img_flat[img_flat < T]
        G2 = img_flat[img_flat >= T]

        if len(G1) == 0 or len(G2) == 0:
            break

        m1 = float(np.mean(G1))
        m2 = float(np.mean(G2))
        T_new = (m1 + m2) / 2

        if abs(T_new - T) < tolerance:
            break
        T = T_new

    binary = np.where(img >= T, 255, 0).astype(np.uint8)
    return int(T), binary


def otsu_threshold(img: np.ndarray) -> tuple:
    """Otsu's method (maximum between-class variance).
    Returns: (threshold, binary_image)
    """
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
    hist = np.histogram(img.ravel(), bins=256, range=(0, 256))[0]

    total = img.size

    sum_total = np.sum(np.arange(256) * hist)

    sum_background = 0
    weight_background = 0

    max_variance = 0
    threshold = 0

    # Evaluate all possible thresholds
    for t in range(256):

        weight_background += hist[t]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background

        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground

        variance_between = (
            weight_background *
            weight_foreground *
            (mean_background - mean_foreground) ** 2
        )

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    binary = np.where(img > threshold, 255, 0).astype(np.uint8)

    return threshold, binary


def otsu_rgb_segmentation(img: np.ndarray) -> np.ndarray:
    """RGB segmentation by applying Otsu per channel.
    Returns 3-channel image where each channel is 0 or 255 (8 colors max).
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = np.stack([img, img, img], axis=2)

    t_R, _ = otsu_threshold(img[:, :, 0])
    t_G, _ = otsu_threshold(img[:, :, 1])
    t_B, _ = otsu_threshold(img[:, :, 2])

    R = np.where(img[:, :, 0] >= t_R, 255, 0)
    G = np.where(img[:, :, 1] >= t_G, 255, 0)
    B = np.where(img[:, :, 2] >= t_B, 255, 0)

    return np.stack([R, G, B], axis=2).astype(np.uint8)
