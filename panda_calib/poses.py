"""Sample diverse joint configurations whose wrist camera sees the whole checkerboard."""
import numpy as np
from scipy.spatial.transform import Rotation

from .board import near_mask
from .camera import project
from .ik import solve_ik
from .kinematics import invert

READY = np.array([0., -.785, 0., -2.356, 0., 1.571, .785])


def look_at_pose(position, target, roll, up_hint=np.array([1., 0., 0.])):
    """Camera (OpenCV axes: z forward, y down) at ``position`` looking at ``target``."""
    z = target-position; z /= np.linalg.norm(z)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)@Rotation.from_rotvec([0., 0., roll]).as_matrix()
    T[:3, 3] = position
    return T


def visible_corners(T_base_cam, T_base_board, board, intrinsics, image_size, mask=None, margin=25.):
    """Boolean (K,) of corners predicted inside the image, in front, and clear of the gripper mask."""
    T_cam_board = invert(T_base_cam)@T_base_board
    pts = board.object_points@T_cam_board[:3, :3].T+T_cam_board[:3, 3]
    if T_cam_board[2, 2] > -.2:  # board seen edge-on or from behind
        return np.zeros(len(pts), bool)
    uv = project(pts, intrinsics)
    w, h = image_size
    ok = (pts[:, 2] > .1)&(uv[:, 0] > margin)&(uv[:, 0] < w-margin)&(uv[:, 1] > margin)&(uv[:, 1] < h-margin)
    if mask is not None:
        ok &= ~near_mask(uv, mask, margin)
    return ok


def board_visible(T_base_cam, T_base_board, board, intrinsics, image_size, mask=None, margin=25.):
    """Checkerboards need every corner; ChArUco boards ``board.min_corners`` (spread over the board)."""
    ok = visible_corners(T_base_cam, T_base_board, board, intrinsics, image_size, mask, margin)
    if board.min_corners is None:
        return bool(ok.all())
    if ok.sum() < board.min_corners:
        return False
    span = np.ptp(board.object_points[ok, :2], axis=0)/np.ptp(board.object_points[:, :2], axis=0)
    return bool(span.min() > .5)


def sample_configurations(model, board, T_base_board, T_hand_cam, intrinsics, image_size, count, rng,
                          distance=(.32, .7), max_tilt_deg=45., roll_deg=90., clearance=.06,
                          q_ref_spread=.6, max_attempts=20000, mask=None):
    """Return (q (count,7), T_base_cam_nominal (count,4,4)) sampled with the nominal model.

    ``mask`` (H,W bool, True = gripper) removes corners hidden by the gripper from the
    visibility test, so planned views keep the board clear of it.
    """
    centre = T_base_board@np.r_[board.object_points.mean(0), 1.]
    normal = T_base_board[:3, 2]
    half = board.object_points.max(0)/2
    qs, poses = [], []
    for _ in range(max_attempts):
        if len(qs) == count:
            break
        tilt = np.deg2rad(max_tilt_deg)*np.sqrt(rng.uniform())
        azimuth = rng.uniform(-np.pi, np.pi)
        direction = Rotation.from_rotvec(normal*azimuth).apply(
            Rotation.from_rotvec(T_base_board[:3, 0]*tilt).apply(normal))
        aim = centre[:3]+T_base_board[:3, :2]@(rng.uniform(-.35, .35, 2)*half[:2])
        position = aim+rng.uniform(*distance)*direction
        # Keep the image "up" roughly away from the robot base so wrist rolls stay feasible.
        up_hint = np.r_[position[:2], 0.]/max(np.linalg.norm(position[:2]), 1e-9)
        T_cam = look_at_pose(position, aim, np.deg2rad(rng.uniform(-roll_deg, roll_deg)), up_hint)
        if not board_visible(T_cam, T_base_board, board, intrinsics, image_size, mask):
            continue
        q_ref = np.clip(READY+rng.normal(0., q_ref_spread, 7), model.lower+.05, model.upper-.05)
        q_ref[0] = np.arctan2(position[1], position[0])+rng.normal(0., .3)
        q, ok, _ = solve_ik(model, T_cam@invert(T_hand_cam), q_ref)
        if not ok or np.any(q < model.lower+.03) or np.any(q > model.upper-.03):
            continue
        _, links = model.link_frames(q[None])
        board_height = T_base_board[2, 3]
        if np.any(links[0, 1:, 2, 3] < board_height+clearance):
            continue
        qs.append(q); poses.append(T_cam)
    if len(qs) < count:
        raise RuntimeError(f'only {len(qs)} of {count} configurations found')
    return np.stack(qs), np.stack(poses)
