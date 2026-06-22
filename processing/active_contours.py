import numpy as np


def inicializar_contornos(rect, shape):
    H, W = shape
    r0, c0, r1, c1 = rect
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H - 1, r1), min(W - 1, c1)

    phi = np.full((H, W), 3, dtype=np.int8)
    phi[r0:r1 + 1, c0:c1 + 1] = -3

    L_in = []
    L_out = []

    for j in range(c0, c1 + 1):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = r0 + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and phi[ni, nj] == 3:
                if (r0, j) not in L_in:
                    L_in.append((r0, j))
                    phi[r0, j] = -1
                break
        if r1 != r0:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = r1 + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and phi[ni, nj] == 3:
                    if (r1, j) not in L_in:
                        L_in.append((r1, j))
                        phi[r1, j] = -1
                    break
    for i in range(r0 + 1, r1):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, c0 + dj
            if 0 <= ni < H and 0 <= nj < W and phi[ni, nj] == 3:
                if (i, c0) not in L_in:
                    L_in.append((i, c0))
                    phi[i, c0] = -1
                break
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, c1 + dj
            if 0 <= ni < H and 0 <= nj < W and phi[ni, nj] == 3:
                if (i, c1) not in L_in:
                    L_in.append((i, c1))
                    phi[i, c1] = -1
                break

    out_set = set()
    for i in range(max(0, r0 - 1), min(H, r1 + 2)):
        for j in range(max(0, c0 - 1), min(W, c1 + 2)):
            if not (r0 <= i <= r1 and c0 <= j <= c1):
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and phi[ni, nj] == -3:
                        out_set.add((i, j))
                        break
    L_out = list(out_set)
    for (i, j) in L_out:
        phi[i, j] = 1

    if not L_out:
        expanded = set()
        for (i, j) in L_in:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not (
                        r0 <= ni <= r1 and c0 <= nj <= c1):
                    expanded.add((ni, nj))
        L_out = list(expanded)
        for (i, j) in L_out:
            phi[i, j] = 1

    return L_in, L_out, phi


def calcular_Fd(imagen, theta0, theta1):
    img = imagen.astype(np.float64)
    if img.ndim == 2:
        dist0 = np.abs(theta0 - img)
        dist1 = np.abs(theta1 - img)
    else:
        t0 = np.array(theta0).reshape(1, 1, -1)
        t1 = np.array(theta1).reshape(1, 1, -1)
        diff0 = img - t0
        diff1 = img - t1
        dist0 = np.linalg.norm(diff0, axis=2)
        dist1 = np.linalg.norm(diff1, axis=2)
    eps = 1e-8
    dist1 = np.maximum(dist1, eps)
    Fd = np.log(dist0 / dist1)
    return Fd


def actualizar_contornos(L_in, L_out, phi, Fd):
    H, W = phi.shape
    set_in = set(L_in)
    set_out = set(L_out)

    to_remove_out = []
    to_add_in = []
    to_add_out = []
    for (i, j) in list(set_out):
        if Fd[i, j] > 0:
            to_remove_out.append((i, j))
            to_add_in.append((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if phi[ni, nj] == 3:
                        to_add_out.append((ni, nj))

    for p in to_remove_out:
        if p in set_out:
            set_out.remove(p)
    for p in to_add_in:
        set_in.add(p)
        phi[p[0], p[1]] = -1
    for p in to_add_out:
        if p not in set_out and phi[p[0], p[1]] == 3:
            set_out.add(p)
            phi[p[0], p[1]] = 1

    to_remove_in_clean = []
    for (i, j) in list(set_in):
        has_positive = False
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if phi[ni, nj] > 0:
                    has_positive = True
                    break
        if not has_positive:
            to_remove_in_clean.append((i, j))
    for p in to_remove_in_clean:
        if p in set_in:
            set_in.remove(p)
            phi[p[0], p[1]] = -3

    to_remove_in = []
    to_add_out2 = []
    to_add_in2 = []
    for (i, j) in list(set_in):
        if Fd[i, j] < 0:
            to_remove_in.append((i, j))
            to_add_out2.append((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if phi[ni, nj] == -3:
                        to_add_in2.append((ni, nj))

    for p in to_remove_in:
        if p in set_in:
            set_in.remove(p)
    for p in to_add_out2:
        set_out.add(p)
        phi[p[0], p[1]] = 1
    for p in to_add_in2:
        if p not in set_in and phi[p[0], p[1]] == -3:
            set_in.add(p)
            phi[p[0], p[1]] = -1

    to_remove_out_clean = []
    for (i, j) in list(set_out):
        has_negative = False
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if phi[ni, nj] < 0:
                    has_negative = True
                    break
        if not has_negative:
            to_remove_out_clean.append((i, j))
    for p in to_remove_out_clean:
        if p in set_out:
            set_out.remove(p)
            phi[p[0], p[1]] = 3

    L_in = list(set_in)
    L_out = list(set_out)
    return L_in, L_out, phi


def verificar_convergencia(L_in, L_out, Fd):
    if not L_in or not L_out:
        return False
    all_out_neg = True
    for (i, j) in L_out:
        if Fd[i, j] >= 0:
            all_out_neg = False
            break
    if all_out_neg:
        return True
    all_in_pos = True
    for (i, j) in L_in:
        if Fd[i, j] <= 0:
            all_in_pos = False
            break
    if all_in_pos:
        return True
    return False


def segmentar_imagen(imagen, rect_inicial, theta0, theta1, max_iter=100):
    H, W = imagen.shape[:2]
    L_in, L_out, phi = inicializar_contornos(rect_inicial, (H, W))
    Fd = calcular_Fd(imagen, theta0, theta1)
    for it in range(max_iter):
        prev_in = set(L_in)
        prev_out = set(L_out)
        L_in, L_out, phi = actualizar_contornos(L_in, L_out, phi, Fd)
        if verificar_convergencia(L_in, L_out, Fd):
            break
        if set(L_in) == prev_in and set(L_out) == prev_out:
            break
    mascara = (phi < 0).astype(np.uint8) * 255
    return mascara, phi, L_in, L_out, it + 1


def estimar_parametros(imagen, rect_inicial, borde_exterior=5):
    H, W = imagen.shape[:2]
    r0, c0, r1, c1 = rect_inicial
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H - 1, r1), min(W - 1, c1)

    interior = imagen[r0:r1 + 1, c0:c1 + 1]
    theta1 = (np.mean(interior, axis=(0, 1))
              if interior.ndim > 2 else np.mean(interior))

    inside_mask = np.zeros((H, W), dtype=bool)
    inside_mask[r0:r1 + 1, c0:c1 + 1] = True

    band_mask = np.zeros((H, W), dtype=bool)
    for di in range(-borde_exterior, borde_exterior + 1):
        for dj in range(-borde_exterior, borde_exterior + 1):
            if di == 0 and dj == 0:
                continue
            r_start = max(0, r0 + di)
            r_end = min(H, r1 + 1 + di)
            c_start = max(0, c0 + dj)
            c_end = min(W, c1 + 1 + dj)
            if r_start < r_end and c_start < c_end:
                band_mask[r_start:r_end, c_start:c_end] = True

    exterior_mask = band_mask & ~inside_mask

    if imagen.ndim == 2:
        exterior_vals = imagen[exterior_mask]
    else:
        exterior_vals = imagen[exterior_mask, :]

    if len(exterior_vals) == 0:
        exterior_mask = ~inside_mask
        if imagen.ndim == 2:
            exterior_vals = imagen[exterior_mask]
        else:
            exterior_vals = imagen[exterior_mask, :]

    theta0 = (np.mean(exterior_vals, axis=0)
              if exterior_vals.ndim > 1 else np.mean(exterior_vals))

    if np.allclose(theta0, theta1, atol=1e-3):
        if isinstance(theta1, np.ndarray):
            theta1 = theta1 + 0.1
        else:
            theta1 += 0.1

    return theta0, theta1
