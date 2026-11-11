"""Gripper occlusion mask for a wrist camera.

The camera is rigid on the hand and the fingers are held at a fixed opening, so the
gripper occupies the same pixels in every calibration image while the scene behind it
changes.  A pixel is static if even its 90th-percentile deviation from the median over
many views is small, i.e. the high-contrast board never passed through it.  Static
regions connected to the image border are taken as the gripper.  Uniform background
that the board never covered is masked too, which costs nothing: no corner was there.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def estimate_static_mask(images, threshold=25., percentile=90., min_area=.002, dilate=4):
    """Mask (H,W bool, True = gripper) from a list of views (RGB or gray uint8).

    ``threshold`` (grey levels) bounds the ``percentile``-th absolute deviation from the
    per-pixel median for a static pixel; it must exceed the gripper's own shading changes
    and stay well below the board's black/white contrast.  Use 20+ views with the board
    spread over the image.
    """
    gray = np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im for im in images]).astype(np.float32)
    if len(gray) < 5:
        raise ValueError('need at least 5 views to separate the gripper from the scene')
    med = np.median(gray, axis=0)
    spread = np.percentile(np.abs(gray-med), percentile, axis=0)
    static = (spread < threshold).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    h, w = static.shape
    mask = np.zeros((h, w), bool)
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        touches_border = x == 0 or y == 0 or x+bw == w or y+bh == h
        if touches_border and area > min_area*h*w:
            mask |= labels == k
    if dilate:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2*dilate+1, 2*dilate+1), np.uint8)) > 0
    return mask


def load_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def save_mask(mask, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)*255).save(path)


def blank_masked(image, mask, margin=4):
    """Fill the occluded area with the median scene grey so it cannot produce corners."""
    out = image.copy()
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((2*margin+1, 2*margin+1), np.uint8)) > 0
    out[grown] = np.median(image[~grown], axis=0).astype(image.dtype)
    return out


def mask_iou(a, b):
    return float((a & b).sum()/max((a | b).sum(), 1))
