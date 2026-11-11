"""Checkerboard geometry and orientation-unambiguous corner detection.

Board frame: origin at inner corner (0, 0), x along a row of ``cols`` corners,
y along the ``rows`` direction, z = x cross y (towards the viewer).  The square
diagonally outside corner (0, 0) is black.  The board must have one even and one
odd number of squares per side so that this corner square is unique up to the
mirror labelling, which the handedness test rejects.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Checkerboard:
    cols: int = 11      # inner corners per row
    rows: int = 8       # inner corners per column
    square: float = .03  # metres

    def __post_init__(self):
        if (self.cols+self.rows) % 2 == 0:
            raise ValueError('use a board with an odd total of inner corners (e.g. 11x8) to fix the orientation')

    @property
    def object_points(self):
        j, i = np.mgrid[:self.rows, :self.cols]
        return np.stack([i.ravel()*self.square, j.ravel()*self.square, np.zeros(i.size)], axis=1)

    def square_is_black(self, i, j):
        """Square (i, j) spans [(i-1)s, i s] x [(j-1)s, j s]; i in 0..cols, j in 0..rows."""
        return (i+j) % 2 == 0

    def detect(self, image):
        """Return (rows*cols, 2) corners in board order, or None when not found."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCornersSB(
            gray, (self.cols, self.rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            return None
        grid = corners.reshape(self.rows, self.cols, 2).astype(float)
        return self.orient(grid, gray)

    def orient(self, grid, gray):
        """Pick the labelling with the board z axis towards the camera and black square at (0,0)."""
        candidates = [grid, grid[::-1, ::-1], grid[::-1, :], grid[:, ::-1]]
        chosen = []
        for g in candidates:
            px, py = g[0, 1]-g[0, 0], g[1, 0]-g[0, 0]
            # Image v points down: a right-handed board facing the camera has negative cross product.
            if px[0]*py[1]-px[1]*py[0] >= 0:
                continue
            black = _patch_mean(gray, g[0, 0]-.5*px-.5*py)
            white = _patch_mean(gray, g[0, 0]+.5*px-.5*py)
            if black is not None and white is not None and black < white:
                chosen.append(g)
        if len(chosen) != 1:
            return None
        return chosen[0].reshape(-1, 2)


def _patch_mean(gray, center, radius=3):
    u, v = np.rint(center).astype(int)
    h, w = gray.shape
    if not (radius <= u < w-radius and radius <= v < h-radius):
        return None
    return float(gray[v-radius:v+radius+1, u-radius:u+radius+1].mean())
