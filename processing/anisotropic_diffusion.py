"""Diffusion filters (isotropic and anisotropic)."""
import numpy as np


def isotropic_diffusion(img: np.ndarray, iterations: int = 20,
                        lambda_param: float = 0.25) -> np.ndarray:
    """Isotropic diffusion (heat equation)."""
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    I = img.astype(np.float32)

    for _ in range(iterations):

        DN = np.roll(I, -1, axis=0) - I
        DS = np.roll(I,  1, axis=0) - I
        DE = np.roll(I, -1, axis=1) - I
        DO = np.roll(I,  1, axis=1) - I

        I = I + lambda_param * (DN + DS + DE + DO)

    return np.clip(I, 0, 255).astype(np.uint8)


def _anisotropic_diffusion_gray(img: np.ndarray, iterations: int = 20,
                                  lambda_param: float = 0.25, k: float = 20,
                                  diffusion_type: str = "leclerc") -> np.ndarray:
    """Perona-Malik anisotropic diffusion on a single channel."""
    I = img.astype(np.float32)

    def g(s):
        if diffusion_type == "lorentz":
            return 1 / (1 + (s / k) ** 2)
        return np.exp(-(s / k) ** 2)

    for _ in range(iterations):
        DN = np.roll(I, -1, axis=0) - I
        DS = np.roll(I,  1, axis=0) - I
        DE = np.roll(I, -1, axis=1) - I
        DO = np.roll(I,  1, axis=1) - I

        I = I + lambda_param * (g(np.abs(DN)) * DN + g(np.abs(DS)) * DS +
                                g(np.abs(DE)) * DE + g(np.abs(DO)) * DO)

    return np.clip(I, 0, 255).astype(np.uint8)


def anisotropic_diffusion(img: np.ndarray, iterations: int = 20,
                          lambda_param: float = 0.25, k: float = 20,
                          diffusion_type: str = "leclerc") -> np.ndarray:
    """Perona-Malik anisotropic diffusion (forces grayscale)."""
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return _anisotropic_diffusion_gray(img, iterations, lambda_param, k, diffusion_type)
