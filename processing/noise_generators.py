"""Noise generators for probability distributions.
All functions use NumPy vectorization.
"""
import numpy as np


def generate_gaussian(n_samples: int, mean: float, sigma: float) -> np.ndarray:
    """Genera n_samples con distribución Gaussiana. Apunte pág. 16.
    
    Uses Box-Muller transform or np.random.normal.
    """
    return np.random.normal(mean, sigma, n_samples)


def generate_exponential(n_samples: int, lam: float) -> np.ndarray:
    """Genera n_samples con distribución Exponencial usando transformada inversa. Apunte págs. 11-12.
    
    PDF: f(x) = lambda * exp(-lambda * x) for x >= 0
    CDF: F(x) = 1 - exp(-lambda * x)
    Inverse: x = -ln(1 - u) / lambda where u ~ Uniform(0,1)
    """
    u = np.random.uniform(0, 1, n_samples)
    return -np.log(1 - u) / lam


def generate_rayleigh(n_samples: int, xi: float) -> np.ndarray:
    """Genera n_samples con distribución Rayleigh usando transformada inversa.
    
    PDF: f(x) = (x / xi^2) * exp(-x^2 / (2 * xi^2)) for x >= 0
    CDF: F(x) = 1 - exp(-x^2 / (2 * xi^2))
    Inverse: x = sqrt(-2 * xi^2 * ln(1 - u)) where u ~ Uniform(0,1)
    """
    u = np.random.uniform(0, 1, n_samples)
    return np.sqrt(-2 * xi**2 * np.log(1 - u))


def gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """PDF de Gaussiana."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * sigma**2))


def exponential_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    """PDF de Exponencial."""
    return lam * np.exp(-lam * x)


def rayleigh_pdf(x: np.ndarray, xi: float) -> np.ndarray:
    """PDF de Rayleigh."""
    return (x / xi**2) * np.exp(-x**2 / (2 * xi**2))