"""Automatic thresholding methods: iterative, Otsu, and Otsu per RGB band."""
import numpy as np


def iterative_threshold(img: np.ndarray, delta_T: float = 1.0) -> tuple:
    """Iterative optimal threshold.
    Returns: (threshold, binary_image)
    """
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    img_flat = img.astype(np.float32).ravel()
    T = float(np.mean(img_flat))

    while True:
        below = img_flat[img_flat < T]
        above = img_flat[img_flat >= T]

        if len(below) == 0 or len(above) == 0:
            break

        m1 = float(np.mean(below))
        m2 = float(np.mean(above))
        T_new = (m1 + m2) / 2

        if abs(T_new - T) < delta_T:
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

    hist = np.bincount(img.ravel().astype(np.int32), minlength=256)
    total = hist.sum()
    if total == 0:
        return 128, np.zeros_like(img)

    p = hist / total
    m_G = float(np.sum(np.arange(256) * p))

    best_t = 0
    max_var = 0.0
    P1 = 0.0
    m_t = 0.0

    for t in range(256):
        P1 += p[t]
        if P1 <= 0 or P1 >= 1:
            continue
        m_t += t * p[t]
        var = (m_G * P1 - m_t) ** 2 / (P1 * (1 - P1))
        if var > max_var:
            max_var = var
            best_t = t

    binary = np.where(img >= best_t, 255, 0).astype(np.uint8)
    return best_t, binary


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
