"""Hough Transform for line detection.
Implementado manualmente (sin cv2.HoughLines).
"""
import numpy as np
from processing.advanced_edge_detection import canny_edge_detector


def hough_lines(img: np.ndarray, threshold: int = 70,
                edge_sigma: float = 1.0, edge_low: float = 50,
                edge_high: float = 150):
    """Detect lines using the Hough Transform.

    Steps:
    1. Edge detection (Canny)
    2. Vote accumulator H(rho, theta)
    3. Find peaks in H
    """
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.uint8)

    theta_res = 1.0
    rho_res = 1.0
    
    # Retorna mapa de bordas binário e pontos de borda (y, x)
    edges = canny_edge_detector(img, edge_sigma, edge_low, edge_high)
    edge_points = np.column_stack(np.where(edges > 0))

    # Accumulator
    h, w = img.shape
    D = max(w, h)
    diagonal = int(np.ceil(np.sqrt(2) * D)) # Máxima distância possível (diagonal da imagem)
    thetas_deg = np.arange(0, 180, theta_res) # Discretização de theta (0 a 179 graus)
    thetas = np.deg2rad(thetas_deg) # Converte para radianos
    n_thetas = len(thetas) 

    rhos = np.arange(-diagonal, diagonal + 1, rho_res) # Discretização de rho
    n_rhos = len(rhos)

    accumulator = np.zeros((n_rhos, n_thetas), dtype=np.int64)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for y, x in edge_points:
        for k in range(n_thetas): 
            rho = x * cos_t[k] + y * sin_t[k]
            rho_idx = int(np.round((rho + diagonal) / rho_res))
            if 0 <= rho_idx < n_rhos:
                accumulator[rho_idx, k] += 1 # Votación

    # Encontrar picos no acumulador
    lines = []
    for i in range(n_rhos): # Percorre cada célula do acumulador
        for j in range(n_thetas):
            if accumulator[i, j] >= threshold: # Verifica se o número de votos é suficiente
                # Usa o máximo local para evitar múltiplas detecções próximas
                i_min = max(0, i - 1)
                i_max = min(n_rhos, i + 2)
                j_min = max(0, j - 1)
                j_max = min(n_thetas, j + 2)
                window = accumulator[i_min:i_max, j_min:j_max]
                if accumulator[i, j] == window.max():
                    rho_val = rhos[i] # Converte o índice para o valor real de rho e theta
                    theta_val = thetas[j]
                    lines.append((rho_val, theta_val, int(accumulator[i, j]))) 

    # Ordena as linhas pelo número de votos (maior primeiro)
    lines.sort(key=lambda x: x[2], reverse=True)

    return edges, accumulator, thetas, rhos, lines


def draw_hough_lines(img, lines, color=(0, 255, 0)):
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=2).astype(np.uint8)
    else:
        rgb = img.copy()

    h, w = rgb.shape[:2]

    # Percorre as linhas detectadas e desenha cada uma usando a equação da reta
    for rho, theta, votes in lines:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Reta vertical
        if abs(sin_t) < 1e-6: # Evita divisão por zero para linhas verticais
            x = int(rho / cos_t)
            if 0 <= x < w:
                rgb[:, x] = color
            continue

        # Desenha usando a equação da reta
        for x in range(w):
            y = int((rho - x * cos_t) / sin_t)

            if 0 <= y < h:
                rgb[y, x] = color

    return rgb
