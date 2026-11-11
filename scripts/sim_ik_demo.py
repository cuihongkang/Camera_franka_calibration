"""Closed-loop check in Isaac Sim: see a target with the wrist camera, IK to it, measure where the TCP lands.

For each trial a held-out calibration view supplies the observation: its detected
corners + PnP + the model under test give the board (target) pose in the base frame.
The TCP is then sent to a board corner (2 cm above it, pointing down) with IK on the
same model; the calibrated models also pre-compensate joint compliance.  The
simulated robot is rebuilt with the dataset's injected errors and the TCP position is
read from the physics engine, so the error measured is what a user would see.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset', type=Path)
p.add_argument('--results', type=Path, default=None)
p.add_argument('--trials', type=int, default=60)
p.add_argument('--hover', type=float, default=.02)
p.add_argument('--tcp', type=float, default=.1034, help='TCP offset along panda_hand z (Franka fingertip)')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--settle-steps', type=int, default=600)
p.add_argument('--gpu', type=int, default=0)
a, extra = p.parse_known_args()
sys.argv = [sys.argv[0], *extra]

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': a.gpu, 'physics_gpu': a.gpu, 'multi_gpu': False})

import cv2
import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction

from panda_calib.board import Checkerboard
from panda_calib.camera import unpack_intrinsics
from panda_calib.export import load_result
from panda_calib.ik import compensate_compliance, solve_ik
from panda_calib.kinematics import ARM_JOINTS, PandaModel, link_side_angles, transform
from panda_calib.poses import READY
from sim_robot import inject_errors, set_joint_drives, set_payload, set_urdf_inertia

ROBOT = '/World/Franka'
USD = ROOT/'assets/isaac_5_1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
MODELS = {'handeye': 'nominal kinematics + hand-eye', 'geometry': 'calibrated geometry',
          'full': 'calibrated geometry + compliance'}


def board_estimate(model, cal, q, tau, corners, board):
    """Base -> board from one view, using the model's kinematics, hand-eye and intrinsics."""
    K, dist = unpack_intrinsics(cal['intrinsics'])
    ok, rvec, tvec = cv2.solvePnP(board.object_points, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    T_cam_board = transform(rvec.ravel(), tvec.ravel())
    q_link = link_side_angles(q, tau, cal['compliance'])
    return model.fk(q_link[None], cal['geo'])[0]@cal['T_hand_cam']@T_cam_board


def main():
    results = a.results or ROOT/'results'/a.dataset.name
    truth = json.loads((a.dataset/'truth.json').read_text())
    meta = json.loads((a.dataset/'metadata.json').read_text())
    board = Checkerboard(meta['board']['cols'], meta['board']['rows'], meta['board']['square'])
    data = np.load(a.dataset/'samples.npz')
    corners = np.load(a.dataset/'corners.npz')['corners']
    test = json.loads((results/'summary.json').read_text())['test']
    cals = {m: load_result(results/m/'calibration.json') for m in MODELS}
    nominal = PandaModel.from_urdf()
    loaded = PandaModel.from_urdf(extra_hand_mass=truth['payload_kg'], extra_hand_com=truth['payload_com'])
    T_hand_tcp = transform(translation=[0., 0., a.tcp])
    rng = np.random.default_rng(a.seed)

    world = World(stage_units_in_meters=1., physics_dt=1/240, rendering_dt=1/60)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, ROBOT).GetPrim().GetReferences().AddReference(str(USD))
    for _ in range(3):
        app.update()
    stiffness = np.array(truth['stiffness'])
    inject_errors(stage, ROBOT, np.array(truth['geo']))
    set_joint_drives(stage, ROBOT, stiffness, np.array(truth['damping']))
    stage.GetPrimAtPath(ROBOT).GetAttribute('physxArticulation:enabledSelfCollisions').Set(False)
    set_payload(stage, ROBOT, truth['payload_kg'], truth['payload_com'])
    set_urdf_inertia(stage, ROBOT, ROOT/'configs/panda_arm_hand.urdf')
    robot = world.scene.add(SingleArticulation(ROBOT, name='franka'))
    hand = SingleXFormPrim(f'{ROBOT}/panda_hand')
    world.reset()
    names = robot.dof_names
    arm = np.array([names.index(n) for n in ARM_JOINTS])
    timeline = omni.timeline.get_timeline_interface()
    T_board_true = np.array(truth['T_base_board'])

    def execute(q_cmd):
        q_full = np.zeros(len(names)); q_full[arm] = q_cmd; q_full[[i for i, n in enumerate(names) if 'finger' in n]] = .02
        robot.set_joint_positions(q_full); robot.set_joint_velocities(np.zeros(len(names)))
        robot.apply_action(ArticulationAction(joint_positions=q_full, joint_velocities=np.zeros(len(names))))
        timeline.play()
        for _ in range(a.settle_steps):
            world.step(render=False)
        pos, quat = hand.get_world_pose()
        T = np.eye(4); T[:3, 3] = pos
        T[:3, :3] = np.array(Gf.Matrix3d(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))).T
        return T@T_hand_tcp, robot.get_measured_joint_efforts()[arm]

    trials = []
    for t in range(a.trials):
        view = int(rng.choice(test))
        k = int(rng.integers(len(board.object_points)))
        yaw = rng.uniform(-np.pi/4, np.pi/4)
        point = np.r_[board.object_points[k], 1.]
        # Tool frame in board coordinates: hover above the corner, z pointing into the board.
        T_board_tool = transform([np.pi, 0., 0.], point[:3]+[0., 0., a.hover])@transform([0., 0., yaw])
        target_true = (T_board_true@T_board_tool)[:3, 3]
        row = dict(trial=t, view=view, corner=k)
        for m, cal in cals.items():
            T_base_board = board_estimate(nominal, cal, data['q'][view], data['tau'][view], corners[view], board)
            q_link, ok, _ = solve_ik(nominal, T_base_board@T_board_tool@np.linalg.inv(T_hand_tcp), READY, cal['geo'])
            if m == 'full':
                q_cmd = compensate_compliance(loaded, q_link, cal['compliance'], cal['geo'])
            else:
                q_cmd = q_link
            T_tcp, tau = execute(q_cmd)
            row[m] = dict(ik_ok=ok, error_mm=float(np.linalg.norm(T_tcp[:3, 3]-target_true)*1e3),
                          board_error_mm=float(np.linalg.norm(T_base_board[:3, 3]-T_board_true[:3, 3])*1e3))
            if m == 'full':
                # One refinement with the measured torque instead of the gravity model.
                q_cmd2 = compensate_compliance(loaded, q_link, cal['compliance'], cal['geo'], tau=tau)
                T_tcp2, _ = execute(q_cmd2)
                row['full_measured_tau'] = dict(ik_ok=ok, error_mm=float(np.linalg.norm(T_tcp2[:3, 3]-target_true)*1e3))
        trials.append(row)
        print(t, {m: round(row[m]['error_mm'], 3) for m in list(MODELS)+['full_measured_tau']}, flush=True)

    summary = {}
    for m in list(MODELS)+['full_measured_tau']:
        e = np.array([r[m]['error_mm'] for r in trials])
        summary[m] = dict(label=MODELS.get(m, 'calibrated + compliance from measured torque (1 iteration)'),
                          mean_mm=float(e.mean()), rms_mm=float(np.sqrt((e**2).mean())),
                          p95_mm=float(np.percentile(e, 95)), max_mm=float(e.max()),
                          ik_failures=int(sum(not r[m]['ik_ok'] for r in trials)))
        print(f"{m:18s} mean {e.mean():.3f}  rms {np.sqrt((e**2).mean()):.3f}  p95 {np.percentile(e, 95):.3f}  "
              f"max {e.max():.3f} mm")
    (results/'ik_demo.json').write_text(json.dumps(dict(summary=summary, trials=trials, hover_m=a.hover,
                                                        tcp_m=a.tcp), indent=1))


try:
    main()
finally:
    app.close()
