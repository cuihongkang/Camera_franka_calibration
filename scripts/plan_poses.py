"""Plan joint configurations for checkerboard data collection (simulation or real robot).

Uses only the nominal model plus rough guesses of the board pose and camera mount,
so the same plan can be executed on a real Panda.  Each configuration is checked for
board visibility (with margin, and clear of the gripper if ``--mask`` is given), joint
limits and link clearance above the board.  A plain checkerboard must be fully
visible; a ChArUco board needs ``--min-corners`` corners spanning half the board.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from panda_calib.board import CharucoBoard, Checkerboard, board_meta
from panda_calib.kinematics import PandaModel, transform
from panda_calib.occlusion import load_mask
from panda_calib.poses import sample_configurations
from panda_calib import synthetic as S

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--count', type=int, default=300)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--board-type', choices=['charuco', 'checkerboard'], default='charuco')
p.add_argument('--board', type=int, nargs=2, default=None,
               help='charuco: squares cols rows (default 12 9); checkerboard: inner corners (default 11 8)')
p.add_argument('--square', type=float, default=.03)
p.add_argument('--marker', type=float, default=.022, help='ChArUco marker side, m')
p.add_argument('--dictionary', default='DICT_5X5_100')
p.add_argument('--min-corners', type=int, default=20, help='ChArUco: minimum visible corners per view')
p.add_argument('--mask', type=Path, default=None, help='gripper occlusion mask PNG (white = gripper)')
p.add_argument('--board-xyz', type=float, nargs=3, default=S.BOARD_POSE[:3, 3].tolist(),
               help='approximate base-frame position of the board origin; board lies flat, x along base x')
p.add_argument('--board-yaw-deg', type=float, default=0.)
p.add_argument('--mount-xyz', type=float, nargs=3, default=S.NOMINAL_MOUNT[:3, 3].tolist(),
               help='approximate panda_hand -> camera translation')
p.add_argument('--mount-tilt-deg', type=float, default=-10.,
               help='camera axes = hand axes rotated about hand y (negative tilts towards the gripper)')
p.add_argument('--fx', type=float, default=S.NOMINAL_INTRINSICS[0])
p.add_argument('--image-size', type=int, nargs=2, default=list(S.IMAGE_SIZE))
p.add_argument('--distance', type=float, nargs=2, default=[.32, .7])
p.add_argument('--max-tilt-deg', type=float, default=45.)
a = p.parse_args()

model = PandaModel.from_urdf()
if a.board_type == 'charuco':
    board = CharucoBoard(*(a.board or (12, 9)), a.square, a.marker, a.dictionary, a.min_corners)
else:
    board = Checkerboard(*(a.board or (11, 8)), a.square)
T_base_board = transform([0., 0., np.deg2rad(a.board_yaw_deg)], a.board_xyz)
T_hand_cam = transform([0., np.deg2rad(a.mount_tilt_deg), 0.], a.mount_xyz)
mask = load_mask(a.mask) if a.mask else None
if mask is not None and mask.shape != (a.image_size[1], a.image_size[0]):
    raise ValueError(f'mask is {mask.shape[::-1]}, image size is {a.image_size}')
w, h = a.image_size
intrinsics = np.array([a.fx, a.fx, w/2, h/2, 0., 0., 0., 0., 0.])
rng = np.random.default_rng(a.seed)
q, T_cam = sample_configurations(model, board, T_base_board, T_hand_cam, intrinsics, a.image_size, a.count, rng,
                                 distance=tuple(a.distance), max_tilt_deg=a.max_tilt_deg, mask=mask)
# Order the plan as a short tour (greedy nearest neighbour in joint space) for faster execution.
order, left = [0], set(range(1, len(q)))
while left:
    j = min(left, key=lambda k: np.abs(q[k]-q[order[-1]]).max())
    order.append(j); left.remove(j)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(
    board=board_meta(board), mask=str(a.mask) if a.mask else None, T_base_board_guess=T_base_board.tolist(),
    T_hand_cam_guess=T_hand_cam.tolist(), intrinsics_guess=intrinsics.tolist(), image_size=a.image_size,
    seed=a.seed, joint_order=[f'panda_joint{i}' for i in range(1, 8)],
    q=q[order].tolist()), indent=1))
print(f'wrote {len(q)} configurations to {a.output}')
