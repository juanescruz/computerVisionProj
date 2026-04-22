"""Image contamination with noise.
Applies random noise to a percentage of image pixels.
"""
import numpy as np
from processing.noise_generators import generate_gaussian, generate_exponential, generate_rayleigh


def add_noise(img: np.ndarray, noise_generator_func, percentage: float, 
             noise_params: dict, mode: str) -> np.ndarray:
    """Contamina un porcentaje aleatorio de píxeles con ruido.
    
    Parámetros:
    - img: imagen original en uint8 (0-255).
    - noise_generator_func: función que genera muestras de ruido.
    - percentage: fracción de píxeles a contaminar (0.0 a 1.0).
    - noise_params: diccionario con parámetros para el generador.
    - mode: 'additive' o 'multiplicative'.
    
    Retorna:
    - img_contaminated: imagen contaminada en uint8.
    """
    img_float = img.astype(np.float32)
    total_pixels = img.size
    num_noisy = int(total_pixels * percentage)
    
    if num_noisy == 0:
        return img.astype(np.uint8)
    
    img_flat = img_float.ravel()
    indices = np.random.choice(total_pixels, size=num_noisy, replace=False)
    
    noise_samples = noise_generator_func(num_noisy, **noise_params)
    
    if mode == 'additive':
        img_flat[indices] += noise_samples
    elif mode == 'multiplicative':
        img_flat[indices] *= noise_samples
    
    img_result = np.clip(img_flat, 0, 255).astype(np.uint8)
    return img_result.reshape(img.shape)


def add_gaussian_noise(img: np.ndarray, percentage: float, mean: float, sigma: float) -> np.ndarray:
    """Aplica ruido Gaussiano aditivo. Apunte pág. 17.
    
    Ruido: n = normal(μ, σ)
    Imagen contaminada: I'(x,y) = I(x,y) + n
    """
    return add_noise(img, generate_gaussian, percentage, 
                   {'mean': mean, 'sigma': sigma}, 'additive')


def add_exponential_noise(img: np.ndarray, percentage: float, lam: float) -> np.ndarray:
    """Aplica ruido Exponencial multiplicativo. Apunte pág. 14.
    
    Ruido: n = exp(λ)
    Imagen contaminada: I'(x,y) = I(x,y) * n
    """
    return add_noise(img, generate_exponential, percentage,
                   {'lam': lam}, 'multiplicative')


def add_rayleigh_noise(img: np.ndarray, percentage: float, xi: float) -> np.ndarray:
    """Aplica ruido Rayleigh multiplicativo.
    
    Ruido: n = rayleigh(ξ)
    Imagen contaminada: I'(x,y) = I(x,y) * n
    """
    return add_noise(img, generate_rayleigh, percentage,
                   {'xi': xi}, 'multiplicative')


def add_salt_pepper_noise(img: np.ndarray, p: float) -> np.ndarray:
    """Aplica ruido Sal y Pimienta a la imagen. Apunte pág. 19.
    
    Parámetros:
    - img: imagen original en uint8 (0-255).
    - p: probabilidad de sal (0) y probabilidad de pimienta (255).
         Debe estar en el rango [0, 0.5].
         Densidad total de ruido = 2p.
    
    Retorna:
    - img_contaminated: imagen contaminada en uint8.
    """
    if p < 0 or p > 0.5:
        raise ValueError("p debe estar en rango [0, 0.5]")
    
    result = img.copy()
    rand = np.random.random(img.shape)
    
    mask_salt = rand <= p
    mask_pepper = rand >= 1 - p
    
    result[mask_salt] = 0
    result[mask_pepper] = 255
    
    return result.astype(np.uint8)