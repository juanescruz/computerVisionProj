"""Gradient edge detectors using NumPy."""
import numpy as np
from processing.spatial_filters import gaussian_filter


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


def prewitt_color(img: np.ndarray, method: str = "euclidean") -> np.ndarray:
    """Prewitt edge detector on RGB. Returns RGB image with colored edges.
    
    Each channel processed independently; result stacked as RGB.
    method: 'euclidean' normalizes by global max across channels.
    Falls back to prewitt_operator if grayscale.
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        return prewitt_operator(img)

    mag_R = prewitt_operator(img[:, :, 0])
    mag_G = prewitt_operator(img[:, :, 1])
    mag_B = prewitt_operator(img[:, :, 2])

    if method == "euclidean":
        max_val = max(mag_R.max(), mag_G.max(), mag_B.max())
        if max_val > 0:
            mag_R = (mag_R.astype(np.float32) / max_val * 255).astype(np.uint8)
            mag_G = (mag_G.astype(np.float32) / max_val * 255).astype(np.uint8)
            mag_B = (mag_B.astype(np.float32) / max_val * 255).astype(np.uint8)

    return np.stack([mag_R, mag_G, mag_B], axis=2).astype(np.uint8)


def sobel_color(img: np.ndarray, method: str = "euclidean") -> np.ndarray:
    """Sobel edge detector on RGB. Returns RGB image with colored edges.
    
    Each channel processed independently; result stacked as RGB.
    method: 'euclidean' normalizes by global max across channels.
    Falls back to sobel_operator if grayscale.
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        return sobel_operator(img)

    mag_R = sobel_operator(img[:, :, 0])
    mag_G = sobel_operator(img[:, :, 1])
    mag_B = sobel_operator(img[:, :, 2])

    if method == "euclidean":
        max_val = max(mag_R.max(), mag_G.max(), mag_B.max())
        if max_val > 0:
            mag_R = (mag_R.astype(np.float32) / max_val * 255).astype(np.uint8)
            mag_G = (mag_G.astype(np.float32) / max_val * 255).astype(np.uint8)
            mag_B = (mag_B.astype(np.float32) / max_val * 255).astype(np.uint8)

    return np.stack([mag_R, mag_G, mag_B], axis=2).astype(np.uint8)


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


def _to_gray(img):
    if len(img.shape) == 3:
        return np.mean(img, axis=2).astype(np.uint8)
    return img


def laplacian_zero_crossings(img: np.ndarray) -> np.ndarray:
    img = _to_gray(img)

    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    lap = convolve2d(img, kernel)

    rows, cols = lap.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            center = lap[i, j]

            neighbors = [
                lap[i-1, j],
                lap[i+1, j],
                lap[i, j-1],
                lap[i, j+1]
            ]

            for n in neighbors:

                # verifica troca de sinal
                if center * n < 0:
                    edges[i, j] = 255
                    break

    return edges


def laplacian_with_slope(img: np.ndarray,
                         threshold: float = None) -> np.ndarray:

    img = _to_gray(img)

    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    lap = convolve2d(img, kernel)

    rows, cols = lap.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)

    # threshold automático
    if threshold is None:
        threshold = np.percentile(np.abs(lap), 80) * 0.25

        if threshold < 1:
            threshold = 1.0

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            center = lap[i, j]

            neighbors = [
                lap[i-1, j],
                lap[i+1, j],
                lap[i, j-1],
                lap[i, j+1]
            ]

            for n in neighbors:

                # verifica troca de sinal
                sign_change = center * n < 0

                # calcula slope
                slope = abs(center - n)

                if sign_change and slope > threshold:
                    edges[i, j] = 255
                    break

    return edges


def log_edge(img: np.ndarray, sigma: float = 1.0, threshold: float = None) -> np.ndarray:
    """LoG (Marr-Hildreth) edge detector. Umbral automático si threshold=None."""
    img = _to_gray(img)

    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    k = size // 2

    kernel = np.zeros((size, size), dtype=np.float32)
    s2 = sigma * sigma
    s4 = s2 * s2
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            r2 = x * x + y * y
            kernel[x + k, y + k] = ((r2 - 2 * s2) / s4) * np.exp(-r2 / (2 * s2))
    kernel = kernel - kernel.mean()

    log_img = convolve2d(img, kernel)

    if threshold is None:
        abs_log = np.abs(log_img)
        threshold = float(np.percentile(abs_log, 85) * 0.25)
        if threshold < 1:
            threshold = 1.0

    h, w = log_img.shape
    edges = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            v = log_img[i, j]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                nv = log_img[ni, nj]
                if (v > 0 and nv < 0) or (v < 0 and nv > 0):
                    slope = abs(v + nv)
                    if slope > threshold:
                        edges[i, j] = 255
                    break
    return edges