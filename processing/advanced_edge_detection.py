import numpy as np
from processing.spatial_filters import gaussian_filter
from processing.edge_detection import convolve2d

def _canny_gaussian_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing for Canny."""
    return gaussian_filter(img, sigma)


def _canny_gradient(img: np.ndarray):
    """Calcula la magnitud y el ángulo del gradiente utilizando los operadores de Sobel."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(img, kernel_x)
    gy = convolve2d(img, kernel_y)

    mag = np.sqrt(gx**2 + gy**2)
    angulo = np.arctan2(gy, gx) * 180.0 / np.pi
    angulo[angulo < 0] += 180.0  # map to [0, 180)

    return mag, angulo


def _non_max_suppression(mag: np.ndarray, angulo: np.ndarray) -> np.ndarray:
    """Suprime los no máximos, conservando únicamente los máximos locales en la dirección del gradiente."""
    h, w = mag.shape
    suppressed = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angulo[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):  # horizontal
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif 22.5 <= a < 67.5:                      # diagonal /
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:                     # vertical
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:                                       # diagonal \
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= n1 and mag[i, j] >= n2: # borrando bordes no maximos
                suppressed[i, j] = mag[i, j]

    return suppressed


def _double_threshold(img: np.ndarray, low: float, high: float):
    """Clasifica los píxeles en bordes fuertes, bordes débiles y píxeles suprimidos."""
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    result = np.zeros_like(img, dtype=np.uint8)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, strong, weak


def _hysteresis(img: np.ndarray, strong: int, weak: int) -> np.ndarray:
    """Conserva los bordes débiles únicamente si están conectados a un borde fuerte (conectividad de 8 vecinos). 
    Utiliza BFS para propagar desde los píxeles fuertes."""
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    mask_fuerte = (img == strong)
    mask_debil = (img == weak)

    # start BFS from all strong pixels
    visited = np.zeros((h, w), dtype=bool)
    queue = list(zip(*np.where(mask_fuerte)))

    for (i, j) in queue:
        visited[i, j] = True

    # 8-connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    while queue:
        i, j = queue.pop(0)
        result[i, j] = 255

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and mask_debil[ni, nj]:
                visited[ni, nj] = True
                queue.append((ni, nj))

    return result


def canny_edge_detector(img: np.ndarray, sigma: float = 1.0,
                        low_threshold: float = 50,
                        high_threshold: float = 150) -> np.ndarray:
    """Detector de bordes Canny implementado manualmente.

    Etapas:
    1. Suavizado Gaussiano (reducción de ruido)
    2. Cálculo del gradiente (Sobel)
    3. Supresión de no máximos
    4. Umbralización doble
    5. Seguimiento de bordes por histéresis

    Parámetros:
    - img: imagen en escala de grises (uint8)
    - sigma: valor sigma del filtro Gaussiano para la reducción de ruido
    - low_threshold: umbral inferior para la histéresis
    - high_threshold: umbral superior para la histéresis

    Retorna:
    - imagen binaria de bordes (uint8, 0 o 255)
    """
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)

    suavizada = _canny_gaussian_smooth(img, sigma)
    mag, angulo = _canny_gradient(suavizada)
    suprimida = _non_max_suppression(mag, angulo)
    umbralizada, val_fuerte, val_debil = _double_threshold(suprimida, low_threshold, high_threshold)
    bordes_final = _hysteresis(umbralizada, val_fuerte, val_debil)

    return bordes_final



# SUSAN

_SUSAN_MASK = np.array([
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
], dtype=bool)

_SUSAN_MASK_SIZE = int(np.sum(_SUSAN_MASK))  # 37

# Desplazamientos (dy, dx) de los 36 píxeles de la máscara circular SUSAN
# relativos al píxel central (núcleo).
# Ejemplo: (-3, 0) representa 3 filas por encima del núcleo.
_OFFSETS = [
    (-3, -1), (-3, 0), (-3, 1),

    (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),

    (-1, -3), (-1, -2), (-1, -1), (-1, 0),
    (-1, 1), (-1, 2), (-1, 3),

    (0, -3), (0, -2), (0, -1),
    (0, 1), (0, 2), (0, 3),

    (1, -3), (1, -2), (1, -1), (1, 0),
    (1, 1), (1, 2), (1, 3),

    (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),

    (3, -1), (3, 0), (3, 1)
]


def _susan_response(img: np.ndarray, t: float = 15) -> np.ndarray:
    """Calcula la respuesta SUSAN s(r0) para cada píxel.
    s(r0) = 1 - n(r0) / 37, donde n(r0) es la suma de c(r, r0)
    sobre la máscara circular.

    c(r, r0) = 1 si |I(r) - I(r0)| < t, en caso contrario 0.

    Retorna una matriz de tipo float con valores en el intervalo [0, 1].
    """
    h, w = img.shape
    img = img.astype(np.int16)
    s = np.zeros((h, w), dtype=np.float64) # Array de respuesta SUSAN

    offsets = _OFFSETS

    for i in range(3, h - 3):
        for j in range(3, w - 3):

            nucleus = img[i, j]
            n = 0

            # Cuenta cuántos vecinos tienen una intensidad similar al núcleo
            for dy, dx in offsets:
                neighbor = img[i + dy, j + dx] # Calcula la posición del vecino utilizando los desplazamientos

                if abs(neighbor - nucleus) < t: 
                    n += 1

            # Respuesta SUSAN (borde, esquina o fondo)
            s[i, j] = 1.0 - (n / _SUSAN_MASK_SIZE)

    return s


def susan_edge_detector(img: np.ndarray, t: float = 15) -> np.ndarray:
    """SUSAN edge detector.
    
    Retorna una imagen binaria de bordes (uint8, 0 o 255).
    Los píxeles de borde cumplen:
    0.35 < s(r0) < 0.60 (aproximadamente s(r0) ≈ 0.5).
    """
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)

    s = _susan_response(img, t)

    edges = np.zeros_like(img, dtype=np.uint8)
    mask = (s > 0.35) & (s < 0.60)
    edges[mask] = 255
    return edges


def susan_corner_detector(img: np.ndarray, t: float = 15) -> np.ndarray:
    """SUSAN corner detector with non-maximum suppression.
    
    Retorna una imagen binaria de esquinas (uint8, 0 o 255).
    Los píxeles de esquina cumplen:
    s(r0) > corner_threshold y son máximos locales en una ventana 3x3.
    """
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)

    s = _susan_response(img, t)

    h, w = s.shape
    corners = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if 0.60 <= s[i, j]:
                window = s[i-1:i+2, j-1:j+2] # Extrae la ventana 3x3 alrededor del píxel actual

                if s[i, j] == np.max(window): # Verifica si el valor en el centro es el máximo local para marcarlo como esquina
                    corners[i, j] = 255

    return corners


def susan_detector(img: np.ndarray, t: float = 15):
    edges = susan_edge_detector(img, t)
    corners = susan_corner_detector(img, t)
    return edges, corners