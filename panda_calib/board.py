"""Calibration targets: plain checkerboard and ChArUco, with gripper-occlusion handling.

Both boards share the interface used by the estimator:

  object_points   (K,3) corner coordinates in the board frame, z = 0 plane
  detect(image, mask=None) -> (K,2) pixels with NaN for corners not observed, or None

The board frame has z pointing towards the viewer.  ``mask`` is a boolean image that
is True where the gripper (or anything rigidly attached to the camera) occludes the
view; corners closer than ``mask_margin`` pixels to it are discarded because the
occluder edge can bias or fake them.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Checkerboard:
    """Plain checkerboard: needs every corner visible.

    Frame: origin at inner corner (0, 0), x along a row of ``cols`` corners, y along
    the ``rows`` direction.  The square diagonally outside corner (0, 0) is black.
    The board must have one even and one odd number of squares per side so that this
    corner square is unique up to the mirror labelling, which the handedness test
    rejects.
    """
    cols: int = 11      # inner corners per row
    rows: int = 8       # inner corners per column
    square: float = .03  # metres
    mask_margin: float = 8.

    kind = 'checkerboard'
    min_corners = None  # all

    def __post_init__(self):
        if (self.cols+self.rows) % 2 == 0:
            raise ValueError('use a board with an odd total of inner corners (e.g. 11x8) to fix the orientation')

    @property
    def object_points(self):
        j, i = np.mgrid[:self.rows, :self.cols]
        return np.stack([i.ravel()*self.square, j.ravel()*self.square, np.zeros(i.size)], axis=1)

    @property
    def outline(self):
        """Board-frame polygon of the printed area (for planning and rendering)."""
        s = self.square
        return np.array([[-s, -s], [self.cols*s, -s], [self.cols*s, self.rows*s], [-s, self.rows*s]])

    def square_is_black(self, i, j):
        """Square (i, j) spans [(i-1)s, i s] x [(j-1)s, j s]; i in 0..cols, j in 0..rows."""
        return (i+j) % 2 == 0

    def detect(self, image, mask=None):
        gray = _gray(image)
        if mask is not None:
            gray = _blank(gray, mask)
        found, corners = cv2.findChessboardCornersSB(
            gray, (self.cols, self.rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            return None
        grid = corners.reshape(self.rows, self.cols, 2).astype(float)
        corners = self.orient(grid, gray)
        if corners is None or (mask is not None and np.any(near_mask(corners, mask, self.mask_margin))):
            return None  # an occluded corner means the grid may be mislabelled
        return corners

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


@dataclass(frozen=True)
class CharucoBoard:
    """ChArUco board: every corner has an id, so partly occluded views are usable.

    Frame: origin at the bottom-left corner of the printed squares when the board
    image from ``image()`` is viewed upright, x to the right, y up, z towards the
    viewer.  (OpenCV's own frame has y down; ``object_points`` converts.)
    """
    cols: int = 12        # squares along x
    rows: int = 9         # squares along y
    square: float = .03
    marker: float = .022
    dictionary: str = 'DICT_5X5_100'
    min_corners: int = 12
    mask_margin: float = 8.

    kind = 'charuco'

    @property
    def cv_board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.cols, self.rows), self.square, self.marker, d)

    @property
    def object_points(self):
        p = self.cv_board.getChessboardCorners().astype(float)
        return np.stack([p[:, 0], self.rows*self.square-p[:, 1], np.zeros(len(p))], axis=1)

    @property
    def outline(self):
        w, h = self.cols*self.square, self.rows*self.square
        return np.array([[0., 0.], [w, 0.], [w, h], [0., h]])

    def image(self, pixels_per_square=200):
        """Printable / texture image covering exactly the squares (no margin)."""
        n = pixels_per_square
        return self.cv_board.generateImage((self.cols*n, self.rows*n), marginSize=0, borderBits=1)

    def detect(self, image, mask=None):
        gray = _gray(image)
        if mask is not None:
            gray = _blank(gray, mask)
        corners, ids, _, _ = cv2.aruco.CharucoDetector(self.cv_board).detectBoard(gray)
        if ids is None or len(ids) == 0:
            return None
        corners = corners.reshape(-1, 2).astype(np.float32)
        if len(corners) >= 2:
            # The detector's default refinement window is fixed; scale it to the square size.
            d = np.linalg.norm(corners[:, None]-corners[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            win = int(np.clip(round(.1*np.median(d.min(axis=1))), 2, 15))
            corners = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2), (win, win), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 100, 1e-4)).reshape(-1, 2)
        out = np.full((len(self.object_points), 2), np.nan)
        out[ids.ravel()] = corners
        if mask is not None:
            out[near_mask(out, mask, self.mask_margin)] = np.nan
        if np.isfinite(out[:, 0]).sum() < self.min_corners:
            return None
        return out


def board_from_meta(meta):
    meta = dict(meta)
    kind = meta.pop('type', 'checkerboard')
    return CharucoBoard(**meta) if kind == 'charuco' else Checkerboard(**meta)


def board_meta(board):
    if board.kind == 'charuco':
        return dict(type='charuco', cols=board.cols, rows=board.rows, square=board.square, marker=board.marker,
                    dictionary=board.dictionary, min_corners=board.min_corners)
    return dict(type='checkerboard', cols=board.cols, rows=board.rows, square=board.square)


def _blank(gray, mask):
    from .occlusion import blank_masked
    return blank_masked(gray, mask)


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def near_mask(points, mask, margin):
    """True for points inside the image-space ``mask`` grown by ``margin`` pixels (NaN -> False)."""
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*int(margin)+1, 2*int(margin)+1))) > 0
    h, w = mask.shape
    out = np.zeros(len(points), bool)
    ok = np.isfinite(points).all(axis=1)
    u = np.clip(np.rint(points[ok, 0]).astype(int), 0, w-1)
    v = np.clip(np.rint(points[ok, 1]).astype(int), 0, h-1)
    out[ok] = grown[v, u]
    return out


def _patch_mean(gray, center, radius=3):
    u, v = np.rint(center).astype(int)
    h, w = gray.shape
    if not (radius <= u < w-radius and radius <= v < h-radius):
        return None
    return float(gray[v-radius:v+radius+1, u-radius:u+radius+1].mean())
