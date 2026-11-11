"""OpenCV-compatible pinhole camera with Brown-Conrady distortion."""
import numpy as np

# fx, fy, cx, cy, k1, k2, p1, p2, k3 (OpenCV distortion order)
INTRINSIC_NAMES = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']


def pack_intrinsics(K, dist):
    dist = np.zeros(5) if dist is None else np.ravel(dist)[:5]
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist])


def unpack_intrinsics(x):
    K = np.array([[x[0], 0., x[2]], [0., x[1], x[3]], [0., 0., 1.]])
    return K, np.asarray(x[4:9])


def project(points, intrinsics):
    """Project camera-frame points (...,3) to pixels (...,2)."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics
    x = points[..., 0]/points[..., 2]
    y = points[..., 1]/points[..., 2]
    r2 = x*x+y*y
    radial = 1+r2*(k1+r2*(k2+r2*k3))
    xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
    return np.stack([fx*xd+cx, fy*yd+cy], axis=-1)
