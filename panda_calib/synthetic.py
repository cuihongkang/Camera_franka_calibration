"""Synthetic ground truth for testing the estimator without rendering."""
import numpy as np

from .camera import project
from .kinematics import invert, transform

DEFAULT_STIFFNESS = np.array([8000., 6000., 7000., 5000., 3000., 2500., 2000.])  # N m / rad
# panda_hand -> camera (OpenCV axes = hand axes, then tilted 10 deg about hand y towards the
# gripper axis): a wrist camera that sees the fingers in one side of the image.
NOMINAL_MOUNT = transform([0., np.deg2rad(-10.), 0.], [.065, 0., 0.])
NOMINAL_INTRINSICS = np.array([924.4, 924.4, 640., 480., 0., 0., 0., 0., 0.])
IMAGE_SIZE = (1280, 960)
BOARD_POSE = transform(np.zeros(3), [.42, -.15, 0.])  # base -> board inner corner (0,0)


def random_truth(rng, translation_mm=.6, rotation_deg=.15, flange=False, mount_mm=3., mount_deg=1.,
                 nominal_mount=NOMINAL_MOUNT):
    """Per-joint frame errors (N_FRAMES,6), compliance and a perturbed camera mount."""
    geo = np.zeros((8, 6))
    geo[:7, :3] = rng.normal(0., translation_mm*1e-3, (7, 3))
    geo[:7, 3:] = rng.normal(0., np.deg2rad(rotation_deg), (7, 3))
    if flange:
        geo[7, :3] = rng.normal(0., translation_mm*1e-3, 3)
        geo[7, 3:] = rng.normal(0., np.deg2rad(rotation_deg), 3)
    mount = np.asarray(nominal_mount)@transform(rng.normal(0., np.deg2rad(mount_deg), 3), rng.normal(0., mount_mm*1e-3, 3))
    return dict(geo=geo, compliance=1/DEFAULT_STIFFNESS, T_hand_cam=mount, T_base_board=BOARD_POSE,
                intrinsics=NOMINAL_INTRINSICS)


def settle(model, q_cmd, truth, iterations=30):
    """Static equilibrium q_link = q_cmd - c * tau_hold(q_link) on the true geometry."""
    q = np.array(q_cmd, float)
    for _ in range(iterations):
        tau = model.holding_torque(q, truth['geo'])
        q = q_cmd-truth['compliance']*tau
    return q, model.holding_torque(q, truth['geo'])


def observe(model, board, q_cmd, truth, rng, pixel_noise=.15, torque_noise=0.):
    q_link, tau = settle(model, q_cmd, truth)
    T_cam = model.fk(q_link, truth['geo'])@truth['T_hand_cam']
    T_cam_board = invert(T_cam)@truth['T_base_board']
    pts = np.einsum('nij,kj->nki', T_cam_board[:, :3, :3], board.object_points)+T_cam_board[:, None, :3, 3]
    uv = project(pts, truth['intrinsics'])+rng.normal(0., pixel_noise, (len(q_cmd), len(board.object_points), 2))
    return uv, tau+rng.normal(0., torque_noise, tau.shape), T_cam, q_link
