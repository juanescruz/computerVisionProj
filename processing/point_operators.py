"""Point operators for image processing.
All functions implemented using NumPy only (no explicit loops).
"""
import numpy as np


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Aplica corrección gamma. Apunte pág. 8.
    
    I_out = I_in^gamma * 255^(1-gamma)
    """
    img_float = img.astype(np.float64) / 255.0
    result = np.power(img_float, gamma)
    result = result * np.power(255.0, 1 - gamma)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


def negative(img: np.ndarray) -> np.ndarray:
    """Negativo de imagen. Apunte pág. 13.
    
    I_out = 255 - I_in
    """
    return (255 - img).astype(np.uint8)


def calc_histogram(img: np.ndarray) -> np.ndarray:
    """Calcula histograma normalizado. Apunte pág. 18.
    
    Returns array of 256 frequency values (relative).
    """
    img_uint8 = img.astype(np.uint8)
    flat = img_uint8.ravel()
    hist = np.bincount(flat, minlength=256)
    return hist / hist.sum()


def threshold(img: np.ndarray, u: int) -> np.ndarray:
    """Umbralización binaria. Apunte pág. 11.
    
    I_out = 0 if I_in < u else 255
    """
    return np.where(img < u, 0, 255).astype(np.uint8)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Ecualización de histograma. Apunte págs. 22-26.
    
    Fórmula discreta:
    - n_k = histograma[k]
    - cdf(k) = Σ_{i=0}^k n_i
    - s_k = (cdf(k) - cdf_min) / (1 - cdf_min) * 255
    """
    flat = img.ravel()
    hist = np.bincount(flat, minlength=256)
    
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    cdf_max = cdf[-1]
    
    cdf_normalized = (cdf - cdf_min) / (cdf_max - cdf_min)
    s_k = cdf_normalized * 255
    
    result = s_k[flat].astype(np.uint8)
    return result.reshape(img.shape)