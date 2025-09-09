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
    mode = ECC_MODELS.get(model, cv2.MOTION_AFFINE)
    if mode == cv2.MOTION_HOMOGRAPHY:
        W = np.eye(3, dtype=np.float32)
    else:
        W = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    ref_f = ref.astype(np.float32)/255.0
    mov_f = mov.astype(np.float32)/255.0
    ref_mask = mask if mask is not None and mask.shape == ref.shape else None
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
    h, w = ref.shape
    if mode == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(mov, W, (w,h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpPerspective(ref_mask, W, (w,h), flags=cv2.INTER_NEAREST)
    else:
        warped = cv2.warpAffine(mov, W, (w,h), flags=cv2.INTER_LINEAR)
        warp_mask = cv2.warpAffine(ref_mask, W, (w,h), flags=cv2.INTER_NEAREST)
    valid_mask = cv2.bitwise_and(ref_mask, warp_mask)
    valid_mask = (valid_mask>0).astype(np.uint8)
    return success, W, warped, valid_mask

def register_orb(ref: np.ndarray, mov: np.ndarray, model: str="homography",
                 orb_features: int = 4000, match_ratio: float = 0.75) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(int(orb_features))
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(mov, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return register_ecc(ref, mov, model=model)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < match_ratio * n.distance]
    if len(good) < 8:
        return register_ecc(ref, mov, model=model)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    h, w = ref.shape
    if model == "homography":
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)
        if H is None:
            return register_ecc(ref, mov, model=model)
        warped = cv2.warpPerspective(mov, H, (w,h), flags=cv2.INTER_LINEAR)
        valid_mask = cv2.warpPerspective(np.ones_like(mov, dtype=np.uint8)*255, H, (w,h), flags=cv2.INTER_NEAREST)
        valid_mask = (valid_mask>0).astype(np.uint8)
        return True, H, warped, valid_mask
    else:
        M, _ = cv2.estimateAffine2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return register_ecc(ref, mov, model=model)
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
        return True, M, warped, valid_mask

def crop_to_overlap(mask: np.ndarray) -> tuple[int,int,int,int]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0,0,0,0
    x,y,w,h = cv2.boundingRect(coords)
    return x,y,w,h
