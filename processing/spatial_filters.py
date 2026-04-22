"""Spatial domain filters.
All filters implemented manually without cv2 or scipy.
"""
import numpy as np


def manual_convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución 2D manual con padding 'edge'.
    
    Parámetros:
    - img: imagen de entrada uint8.
    - kernel: kernel de convolución float32.
    
    Retorna:
    - imagen convolucionada uint8.
    """
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    img_float = img.astype(np.float32)
    img_padded = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros((img_h, img_w), dtype=np.float32)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = img_padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(roi * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def manual_median_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Filtro de mediana. Apunte pág. 30.
    
    Parámetros:
    - img: imagen de entrada uint8.
    - kernel_size: tamaño del kernel (impar).
    
    Retorna:
    - imagen filtrada uint8.
    """
    img_h, img_w = img.shape
    pad = kernel_size // 2
    
    img_padded = np.pad(img, pad, mode='edge')
    output = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = img_padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(roi)
    
    return output


def mean_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Filtro de media. Apunte pág. 22.
    
    Parámetros:
    - img: imagen de entrada uint8.
    - kernel_size: tamaño del kernel (impar).
    
    Retorna:
    - imagen filtrada uint8.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= (kernel_size * kernel_size)
    return manual_convolution(img, kernel)


def median_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Filtro de mediana. Apunte pág. 30."""
    return manual_median_filter(img, kernel_size)


def weighted_median_filter(img: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Filtro de mediana ponderada. Apunte pág. 34.
    
    Parámetros:
    - img: imagen de entrada uint8.
    - weights: kernel de pesos (default 3x3 con centro 3).
    
    Retorna:
    - imagen filtrada uint8.
    """
    if weights is None:
        weights = np.array([
            [1, 1, 1],
            [1, 3, 1],
            [1, 1, 1]
        ], dtype=np.int32)
    
    kernel_size = weights.shape[0]
    pad = kernel_size // 2
    
    img_padded = np.pad(img, pad, mode='edge')
    img_h, img_w = img.shape
    output = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = img_padded[i:i+kernel_size, j:j+kernel_size]
            
            expanded = []
            for w_i in range(kernel_size):
                for w_j in range(kernel_size):
                    val = roi[w_i, w_j]
                    count = weights[w_i, w_j]
                    expanded.extend([val] * count)
            
            expanded = np.array(expanded)
            output[i, j] = np.median(expanded)
    
    return output


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Genera kernel Gaussiano 2D. Apunte págs. 25-28.
    
    Parámetros:
    - sigma: desviación estándar.
    
    Retorna:
    - kernel Gaussiano normalizado.
    """
    k = int(2 * sigma + 1)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    
    center = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)
    
    for i in range(k):
        for j in range(k):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel /= kernel.sum()
    return kernel


def gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """Filtro Gaussiano. Apunte págs. 25-28.
    
    Parámetros:
    - img: imagen de entrada uint8.
    - sigma: desviación estándar.
    
    Retorna:
    - imagen filtrada uint8.
    """
    kernel = gaussian_kernel(sigma)
    return manual_convolution(img, kernel)


def edge_enhancement_filter(img: np.ndarray) -> np.ndarray:
    """Realce de bordes. Apunte págs. 36-37.
    
    Kernel:
    [-1, -1, -1]
    [-1,  9, -1]
    [-1, -1, -1]
    
    Parámetros:
    - img: imagen de entrada uint8.
    
    Retorna:
    - imagen con bordes realzados uint8.
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    return manual_convolution(img, kernel)