"""Plan joint configurations for checkerboard data collection (simulation or real robot).

Uses only the nominal model plus rough guesses of the board pose and camera mount,
so the same plan can be executed on a real Panda.  Each configuration is checked for
full-board visibility (with margin), joint limits and link clearance above the board.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from panda_calib.board import Checkerboard
from panda_calib.kinematics import PandaModel, transform
from panda_calib.poses import sample_configurations
from panda_calib import synthetic as S

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--count', type=int, default=300)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--board', type=int, nargs=2, default=[11, 8], help='inner corners: cols rows')
p.add_argument('--square', type=float, default=.03)
p.add_argument('--board-xyz', type=float, nargs=3, default=S.BOARD_POSE[:3, 3].tolist(),
               help='approximate base-frame position of inner corner (0,0); board lies flat, x along base x')
p.add_argument('--board-yaw-deg', type=float, default=0.)
p.add_argument('--mount-xyz', type=float, nargs=3, default=S.NOMINAL_MOUNT[:3, 3].tolist(),
               help='approximate panda_hand -> camera translation (camera axes = hand axes)')
p.add_argument('--fx', type=float, default=S.NOMINAL_INTRINSICS[0])
p.add_argument('--image-size', type=int, nargs=2, default=list(S.IMAGE_SIZE))
p.add_argument('--distance', type=float, nargs=2, default=[.32, .7])
p.add_argument('--max-tilt-deg', type=float, default=45.)
a = p.parse_args()

model = PandaModel.from_urdf()
board = Checkerboard(*a.board, a.square)
T_base_board = transform([0., 0., np.deg2rad(a.board_yaw_deg)], a.board_xyz)
T_hand_cam = transform(np.zeros(3), a.mount_xyz)
w, h = a.image_size
intrinsics = np.array([a.fx, a.fx, w/2, h/2, 0., 0., 0., 0., 0.])
rng = np.random.default_rng(a.seed)
q, T_cam = sample_configurations(model, board, T_base_board, T_hand_cam, intrinsics, a.image_size, a.count, rng,
                                 distance=tuple(a.distance), max_tilt_deg=a.max_tilt_deg)
# Order the plan as a short tour (greedy nearest neighbour in joint space) for faster execution.
order, left = [0], set(range(1, len(q)))
while left:
    j = min(left, key=lambda k: np.abs(q[k]-q[order[-1]]).max())
    order.append(j); left.remove(j)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(
    board=dict(cols=board.cols, rows=board.rows, square=board.square), T_base_board_guess=T_base_board.tolist(),
    T_hand_cam_guess=T_hand_cam.tolist(), intrinsics_guess=intrinsics.tolist(), image_size=a.image_size,
    seed=a.seed, joint_order=[f'panda_joint{i}' for i in range(1, 8)],
    q=q[order].tolist()), indent=1))
print(f'wrote {len(q)} configurations to {a.output}')
