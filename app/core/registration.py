from __future__ import annotations
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

ECC_MODELS = {
    'translation': cv2.MOTION_TRANSLATION,
    'euclidean': cv2.MOTION_EUCLIDEAN,
    'affine': cv2.MOTION_AFFINE,
    'homography': cv2.MOTION_HOMOGRAPHY,
}

def has_cuda() -> bool:
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

def preprocess(gray: np.ndarray, gauss_sigma: float, clahe_clip: float, clahe_grid: int) -> np.ndarray:
    g = gray
    if gauss_sigma > 0:
        k = int(max(1, round(gauss_sigma*3))*2 + 1)
        g = cv2.GaussianBlur(g, (k,k), gauss_sigma)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid,clahe_grid))
        g = clahe.apply(g)
    return g

def register_ecc(ref: np.ndarray, mov: np.ndarray, model: str="affine",
                 max_iters: int=1000, eps: float=1e-6,
                 mask: Optional[np.ndarray]=None) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Register *mov* to *ref* using the ECC algorithm.

    The Enhanced Correlation Coefficient (ECC) maximizes image intensity
    agreement, providing sub-pixel accuracy. It is sensitive to frames with
    little variation, which can cause the optimization to diverge and produce
    NaN values in the resulting warp matrix. Such nearly uniform images are
    detected and handled gracefully by returning an identity transform.
    """
    if mov.size == 0 or ref.size == 0:
        logging.warning("Skipping registration: empty frame")
        return False, np.eye(3, dtype=np.float32), mov, np.zeros_like(mov, dtype=np.uint8)
    mode = ECC_MODELS.get(model, cv2.MOTION_AFFINE)
    if mode == cv2.MOTION_HOMOGRAPHY:
        W = np.eye(3, dtype=np.float32)
    else:
        W = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    ref_f = ref.astype(np.float32)/255.0
    mov_f = mov.astype(np.float32)/255.0
    ref_mask = mask if mask is not None and mask.shape == ref.shape else None

    # Check for low-variance images which can cause ECC to fail
    ref_var = float(np.var(ref_f))
    mov_var = float(np.var(mov_f))
    if ref_var < 1e-6 or mov_var < 1e-6:
        logging.warning("Skipping registration: low variance (ref=%e, mov=%e)", ref_var, mov_var)
        identity = np.eye(3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2,3, dtype=np.float32)
        return False, identity, mov, np.zeros_like(mov, dtype=np.uint8)

    success = True
    try:
        if ref_mask is not None:
            _, W = cv2.findTransformECC(ref_f, mov_f, W, mode, criteria, inputMask=ref_mask, gaussFiltSize=5)
        else:
            ref_mask = np.ones_like(ref, dtype=np.uint8)*255
            _, W = cv2.findTransformECC(ref_f, mov_f, W, mode, criteria)
    except cv2.error as e:
        logging.exception("ECC registration failed: %s", e)
        success = False

    if not success or np.isnan(W).any():
        if np.isnan(W).any():
            logging.warning("ECC produced NaN warp matrix; returning identity")
        identity = np.eye(3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2,3, dtype=np.float32)
        return False, identity, mov, np.zeros_like(mov, dtype=np.uint8)

    h, w = ref.shape
    if mode == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(mov, W, (w,h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpPerspective(ref_mask, W, (w,h), flags=cv2.INTER_NEAREST)
    else:
        warped = cv2.warpAffine(mov, W, (w,h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpAffine(ref_mask, W, (w,h), flags=cv2.INTER_NEAREST)
    valid_mask = cv2.bitwise_and(ref_mask, warp_mask)
    valid_mask = (valid_mask>0).astype(np.uint8)
    return True, W, warped, valid_mask

def register_orb(ref: np.ndarray, mov: np.ndarray, model: str="homography",
                 orb_features: int = 4000, match_ratio: float = 0.75,
                 fallback_model: str = "affine", *,
                 min_keypoints: int = 8, min_matches: int = 8,
                 use_ecc_fallback: bool = True) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, bool]:
    if mov.size == 0 or ref.size == 0:
        logging.warning("Skipping registration: empty frame")
        return False, np.eye(3, dtype=np.float32), mov, np.zeros_like(mov, dtype=np.uint8), False
    orb = cv2.ORB_create(int(orb_features))
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(mov, None)
    n1, n2 = len(k1), len(k2)
    if d1 is None or d2 is None or n1 < min_keypoints or n2 < min_keypoints:
        logging.warning(
            "Insufficient ORB features (ref=%d, mov=%d)" + (
                "; falling back to ECC registration" if use_ecc_fallback else ""
            ),
            n1,
            n2,
        )
        if use_ecc_fallback:
            success, W, warped, valid_mask = register_ecc(ref, mov, model=fallback_model)
            return success, W, warped, valid_mask, True
        identity = np.eye(3, dtype=np.float32) if model == "homography" else np.eye(2, 3, dtype=np.float32)
        return False, identity, mov, np.zeros_like(mov, dtype=np.uint8), False
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < match_ratio * n.distance]
    if len(good) < min_matches:
        logging.warning(
            "Too few good ORB matches" + (
                "; falling back to ECC registration" if use_ecc_fallback else ""
            )
        )
        if use_ecc_fallback:
            success, W, warped, valid_mask = register_ecc(ref, mov, model=fallback_model)
            return success, W, warped, valid_mask, True
        identity = np.eye(3, dtype=np.float32) if model == "homography" else np.eye(2, 3, dtype=np.float32)
        return False, identity, mov, np.zeros_like(mov, dtype=np.uint8), False
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    h, w = ref.shape
    if model == "homography":
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)
        if H is None:
            logging.warning("cv2.findHomography failed; falling back to ECC registration")
            success, W, warped, valid_mask = register_ecc(ref, mov, model=fallback_model)
            return success, W, warped, valid_mask, True
        det = float(abs(np.linalg.det(H)))
        inlier_ratio = float(mask.sum()) / float(mask.size) if mask is not None else 0.0
        if det < 1e-6 or inlier_ratio < 0.5:
            logging.warning("Homography validation failed (det=%e, inlier_ratio=%.3f); falling back to ECC", det, inlier_ratio)
            success, W, warped, valid_mask = register_ecc(ref, mov, model=fallback_model)
            return success, W, warped, valid_mask, True
        warped = cv2.warpPerspective(mov, H, (w,h), flags=cv2.INTER_LINEAR)
        valid_mask = cv2.warpPerspective(np.ones_like(mov, dtype=np.uint8)*255, H, (w,h), flags=cv2.INTER_NEAREST)
        valid_mask = (valid_mask>0).astype(np.uint8)
        return True, H, warped, valid_mask, False
    else:
        M, _ = cv2.estimateAffine2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            logging.warning("cv2.estimateAffine2D failed; falling back to ECC registration")
            success, W, warped, valid_mask = register_ecc(ref, mov, model=fallback_model)
            return success, W, warped, valid_mask, True
        if model == "translation":
            M[:,:2] = np.eye(2, dtype=np.float32)
        elif model == "euclidean":
            R = M[:,:2]
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            M[:,:2] = R
        warped = cv2.warpAffine(mov, M, (w,h), flags=cv2.INTER_LINEAR)
        valid_mask = cv2.warpAffine(np.ones_like(mov, dtype=np.uint8)*255, M, (w,h), flags=cv2.INTER_NEAREST)
        valid_mask = (valid_mask>0).astype(np.uint8)
        return True, M, warped, valid_mask, False

def register_orb_ecc(ref: np.ndarray, mov: np.ndarray, model: str = "affine",
                     max_iters: int = 1000, eps: float = 1e-6,
                     orb_features: int = 4000, match_ratio: float = 0.75,
                     fallback_model: str = "affine", *,
                     min_keypoints: int = 8, min_matches: int = 8,
                     use_ecc_fallback: bool = True,
                     mask: Optional[np.ndarray] = None) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Register using ORB for initialization followed by ECC refinement."""
    if mov.size == 0 or ref.size == 0:
        logging.warning("Skipping registration: empty frame")
        return False, np.eye(3, dtype=np.float32), mov, np.zeros_like(mov, dtype=np.uint8)

    # Initial alignment via ORB keypoints
    success, W_orb, warped, valid_mask, fb = register_orb(
        ref, mov, model=model, orb_features=orb_features,
        match_ratio=match_ratio, fallback_model=fallback_model,
        min_keypoints=min_keypoints, min_matches=min_matches,
        use_ecc_fallback=use_ecc_fallback,
    )
    if fb or not success:
        # Either ORB failed or already fell back to ECC; return its result
        return success, W_orb, warped, valid_mask

    mode = ECC_MODELS.get(model, cv2.MOTION_AFFINE)
    W_init = W_orb.astype(np.float32)
    if mode != cv2.MOTION_HOMOGRAPHY and W_init.shape == (3, 3):
        W_init = W_init[:2, :]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    ref_f = ref.astype(np.float32) / 255.0
    mov_f = mov.astype(np.float32) / 255.0
    ref_mask = mask if mask is not None and mask.shape == ref.shape else None

    try:
        if ref_mask is not None:
            _, W = cv2.findTransformECC(ref_f, mov_f, W_init, mode, criteria, inputMask=ref_mask, gaussFiltSize=5)
        else:
            ref_mask = np.ones_like(ref, dtype=np.uint8) * 255
            _, W = cv2.findTransformECC(ref_f, mov_f, W_init, mode, criteria)
    except cv2.error as e:
        logging.exception("ORB+ECC registration failed: %s", e)
        return success, W_orb, warped, valid_mask

    h, w = ref.shape
    if mode == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(mov, W, (w, h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpPerspective(ref_mask, W, (w, h), flags=cv2.INTER_NEAREST)
    else:
        warped = cv2.warpAffine(mov, W, (w, h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpAffine(ref_mask, W, (w, h), flags=cv2.INTER_NEAREST)
    valid_mask = cv2.bitwise_and(ref_mask, warp_mask)
    valid_mask = (valid_mask > 0).astype(np.uint8)
    return True, W, warped, valid_mask

def crop_to_overlap(mask: np.ndarray) -> tuple[int,int,int,int]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0,0,0,0
    x,y,w,h = cv2.boundingRect(coords)
    return x,y,w,h
