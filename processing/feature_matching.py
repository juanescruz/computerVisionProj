"""SIFT feature detection and matching using OpenCV."""
import numpy as np
import cv2

from processing.noise_contamination import add_gaussian_noise


def detect_sift(img: np.ndarray):
    """Detect SIFT keypoints and descriptors.
    
    Parameters:
    - img: uint8 grayscale image
    
    Returns:
    - keypoints: list of cv2.KeyPoint
    - descriptors: numpy array of shape (N, 128)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   ratio_thresh: float = 0.75):
    """Match SIFT descriptors using Lowe's ratio test.
    
    Parameters:
    - desc1: descriptors from image 1 (N, 128)
    - desc2: descriptors from image 2 (M, 128)
    - ratio_thresh: Lowe's ratio threshold for good matches
    
    Returns:
    - good_matches: list of cv2.DMatch passing ratio test
    - all_matches: list of all cv2.DMatch (before ratio test)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    all_matches = []
    good_matches = []
    for pair in knn_matches:
        if len(pair) >= 2:
            m, n = pair[0], pair[1]
            all_matches.append(m)
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches, all_matches


def draw_keypoints(img: np.ndarray, keypoints, color=None) -> np.ndarray:
    """Draw SIFT keypoints on image, return RGB."""
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img.copy()
    return cv2.drawKeypoints(rgb, keypoints, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(img1: np.ndarray, kp1, img2: np.ndarray, kp2,
                 matches, max_matches: int = 50) -> np.ndarray:
    """Draw matching lines between two images."""
    if img1.ndim == 2:
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        rgb1 = img1.copy()
    if img2.ndim == 2:
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        rgb2 = img2.copy()

    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) > max_matches:
        matches = matches[:max_matches]

    return cv2.drawMatches(rgb1, kp1, rgb2, kp2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# ─── Image transformations ───

def apply_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)


def apply_scale(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale image by factor, then resize back to original size."""
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_translation(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Translate image by (tx, ty) pixels."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)


def apply_illumination(img: np.ndarray, alpha: float = 1.0,
                       beta: int = 0) -> np.ndarray:
    """Change brightness/contrast: alpha*img + beta."""
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def apply_perspective(img: np.ndarray, skew: float = 0.2) -> np.ndarray:
    """Apply perspective transform (skew)."""
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    offset = int(w * skew)
    dst_pts = np.float32([[offset, 0], [w - 1 - offset, 0],
                          [0, h - 1], [w - 1, h - 1]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)


def apply_gaussian_noise(img: np.ndarray, mean: float = 0,
                         sigma: float = 25) -> np.ndarray:
    """Add Gaussian noise to image."""
    return add_gaussian_noise(img, percentage=1.0, mean=mean, sigma=sigma)