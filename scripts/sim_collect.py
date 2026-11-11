"""Render eye-in-hand checkerboard views of a Franka with injected joint errors in Isaac Sim 5.1.

The simulated robot is the "real robot": its joint frames carry random 6-DoF errors
and its joint drives act as compliant joints under gravity.  Recorded per view:

  q_cmd   drive targets = what the encoders of a motor-side-sensed joint report
  tau     solver-measured joint efforts (stand-in for Franka tau_J)
  image   RGB from the wrist camera

Ground truth (link-side angles, renderer camera pose, injected errors) is written
separately to truth.json / truth.npz and is never read by the calibration.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan', type=Path, default=ROOT/'configs/poses_sim.json')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--limit', type=int, default=None, help='only the first N planned views')
p.add_argument('--truth-seed', type=int, default=0)
p.add_argument('--translation-mm', type=float, default=.6, help='std of injected joint-frame translation errors')
p.add_argument('--rotation-deg', type=float, default=.15, help='std of injected joint-frame rotation errors')
p.add_argument('--stiffness', type=float, nargs=7, default=None, help='true joint stiffness, N m/rad')
p.add_argument('--payload-kg', type=float, default=.3, help='camera + mount mass added to panda_hand')
p.add_argument('--payload-com', type=float, nargs=3, default=[.065, 0., .03])
p.add_argument('--settle-steps', type=int, default=600)
p.add_argument('--rt-subframes', type=int, default=16)
p.add_argument('--gpu', type=int, default=0)
a, extra = p.parse_known_args()
a.output = a.output.resolve()
if a.output.exists():
    raise FileExistsError(a.output)
sys.argv = [sys.argv[0], *extra]

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': a.gpu, 'physics_gpu': a.gpu, 'multi_gpu': False,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 0, 'width': 1280, 'height': 960})

import numpy as np
from PIL import Image
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera

from panda_calib.board import board_from_meta
from panda_calib.occlusion import save_mask
from panda_calib.kinematics import ARM_JOINTS, PandaModel, invert, transform
from panda_calib import synthetic as S
from sim_robot import build_board, inject_errors, joint_origins, label_gripper, set_joint_drives, set_payload, set_urdf_inertia

ROBOT = '/World/Franka'
USD = ROOT/'assets/isaac_5_1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'


def main():
    plan = json.loads(a.plan.read_text())
    q_plan = np.array(plan['q'])[:a.limit]
    board = board_from_meta(plan['board'])
    rng = np.random.default_rng(a.truth_seed)
    truth = S.random_truth(rng, a.translation_mm, a.rotation_deg, nominal_mount=np.array(plan['T_hand_cam_guess']))
    stiffness = S.DEFAULT_STIFFNESS if a.stiffness is None else np.array(a.stiffness)
    truth['compliance'] = 1/stiffness
    T_world_board = np.array(plan['T_base_board_guess'])@transform(rng.normal(0, np.deg2rad(.5), 3)*[0, 0, 1],
                                                                    rng.normal(0, 3e-3, 3)*[1, 1, 0])
    truth['T_base_board'] = T_world_board
    damping = 2*np.sqrt(stiffness*.5)  # ~critical for 0.5 kg m^2 effective inertia

    (a.output/'images').mkdir(parents=True)
    world = World(stage_units_in_meters=1., physics_dt=1/240, rendering_dt=1/60)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, ROBOT).GetPrim().GetReferences().AddReference(str(USD))
    for _ in range(3):
        app.update()
    model = PandaModel.from_urdf()
    origins = joint_origins(stage, ROBOT)
    origin_check = float(np.abs(origins-model.origins).max())
    if origin_check > 1e-5:
        raise RuntimeError(f'USD joint frames differ from the URDF by {origin_check}')
    inject_errors(stage, ROBOT, truth['geo'])
    set_joint_drives(stage, ROBOT, stiffness, damping)
    stage.GetPrimAtPath(ROBOT).GetAttribute('physxArticulation:enabledSelfCollisions').Set(False)
    set_payload(stage, ROBOT, a.payload_kg, a.payload_com)
    set_urdf_inertia(stage, ROBOT, ROOT/'configs/panda_arm_hand.urdf')
    build_board(stage, board, T_world_board, a.output/'board_texture.png')
    label_gripper(stage, ROBOT)
    dome = UsdLux.DomeLight.Define(stage, '/World/Dome'); dome.CreateIntensityAttr(900.)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun'); sun.CreateIntensityAttr(1500.); sun.CreateAngleAttr(20.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(15., 10., 0.))

    robot = world.scene.add(SingleArticulation(ROBOT, name='franka'))
    hand = SingleXFormPrim(f'{ROBOT}/panda_hand')
    w, h = plan['image_size']
    cam = Camera(f'{ROBOT}/panda_hand/WristCamera', resolution=(w, h))
    mount = truth['T_hand_cam']
    quat = Gf.Matrix4d(mount.T.tolist()).ExtractRotationQuat()
    cam.set_local_pose(translation=mount[:3, 3], orientation=np.array([quat.GetReal(), *quat.GetImaginary()]),
                       camera_axes='ros')
    world.reset()
    cam.initialize()
    fx = plan['intrinsics_guess'][0]
    cam.set_horizontal_aperture(36.); cam.set_vertical_aperture(36.*h/w)
    cam.set_focal_length(fx*36./w); cam.set_clipping_range(.02, 20.)
    params_ann = rep.AnnotatorRegistry.get_annotator('camera_params')
    params_ann.attach([cam.get_render_product_path()])
    seg_ann = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
    seg_ann.attach([cam.get_render_product_path()])
    gripper_union = None
    rep.orchestrator.set_capture_on_play(False)

    names = robot.dof_names
    arm = np.array([names.index(n) for n in ARM_JOINTS])
    fingers = np.array([i for i, n in enumerate(names) if 'finger' in n])
    model_load = PandaModel.from_urdf(extra_hand_mass=a.payload_kg, extra_hand_com=a.payload_com)
    timeline = omni.timeline.get_timeline_interface()
    records = []
    for n, q_cmd in enumerate(q_plan):
        q_full = np.zeros(len(names)); q_full[arm] = q_cmd; q_full[fingers] = .02
        q_guess = q_full.copy()
        q_guess[arm] = S.settle(model_load, q_cmd[None], dict(geo=truth['geo'], compliance=truth['compliance']))[0][0]
        robot.set_joint_positions(q_guess); robot.set_joint_velocities(np.zeros(len(names)))
        robot.apply_action(ArticulationAction(joint_positions=q_full, joint_velocities=np.zeros(len(names))))
        timeline.play(); timeline.commit()
        for k in range(a.settle_steps):
            world.step(render=False)
            if k == a.settle_steps-121:
                q_before = robot.get_joint_positions()[arm]
        q_link = robot.get_joint_positions()[arm]
        qd = q_link-q_before  # position change over the last 0.5 s: convergence check
        tau = robot.get_measured_joint_efforts()[arm]
        timeline.pause(); timeline.commit()
        # Without a few app updates the renderer keeps stale poses for the gripper meshes
        # after headless physics steps (they vanish from the wrist view).
        for _ in range(3):
            app.update()
        rep.orchestrator.step(rt_subframes=a.rt_subframes, delta_time=0., pause_timeline=True)
        view = np.asarray(params_ann.get_data()['cameraViewTransform']).reshape(4, 4)
        T_world_cam = np.linalg.inv(np.diag([1., -1., -1., 1.])@view.T)  # renderer -> OpenCV axes
        pos, quat = hand.get_world_pose()
        T_world_hand = transform(np.zeros(3), pos)
        T_world_hand[:3, :3] = np.array(Gf.Matrix3d(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))).T
        rgb = np.asarray(cam.get_rgba())[..., :3]
        Image.fromarray(rgb).save(a.output/'images'/f'{n:06d}.png')
        seg = seg_ann.get_data()
        ids = [int(k) for k, v in seg['info']['idToLabels'].items() if 'gripper' in str(v)]
        gripper = np.isin(np.asarray(seg['data']).squeeze(), ids)
        gripper_union = gripper if gripper_union is None else gripper_union | gripper
        T_fk = model.fk(q_link[None], truth['geo'])[0]
        records.append(dict(q_cmd=q_cmd, q_link=q_link, tau=tau, qd=qd, T_world_cam=T_world_cam,
                            T_world_hand=T_world_hand, T_fk_hand=T_fk))
        if n % 20 == 0 or n < 3:
            fk_err = np.linalg.norm(T_fk[:3, 3]-T_world_hand[:3, 3])*1e3
            cam_err = np.linalg.norm((T_fk@mount)[:3, 3]-T_world_cam[:3, 3])*1e3
            print(f'view {n}: drift {np.abs(qd).max():.2e} rad  deflection mrad {np.round((q_cmd-q_link)*1e3, 3)}'
                  f'  tau {np.round(tau, 2)}  hand FK err {fk_err:.4f} mm  camera err {cam_err:.4f} mm', flush=True)

    rec = {k: np.stack([r[k] for r in records]) for k in records[0]}
    save_mask(gripper_union, a.output/'truth_gripper_mask.png')
    print(f'gripper covers {gripper_union.mean()*100:.1f}% of the image', flush=True)
    w_, h_ = cam.get_resolution()
    K = cam.get_intrinsics_matrix()
    np.savez(a.output/'samples.npz', q=rec['q_cmd'], tau=rec['tau'], image_size=np.array([w_, h_]))
    np.savez(a.output/'truth.npz', q_link=rec['q_link'], settle_drift=rec['qd'], T_world_cam=rec['T_world_cam'],
             T_world_hand=rec['T_world_hand'], T_fk_hand=rec['T_fk_hand'], K=K)
    (a.output/'truth.json').write_text(json.dumps(dict(
        geo=truth['geo'].tolist(), compliance=truth['compliance'].tolist(), stiffness=stiffness.tolist(),
        damping=damping.tolist(), T_hand_cam=mount.tolist(), T_base_board=T_world_board.tolist(),
        K=np.asarray(K).tolist(), payload_kg=a.payload_kg, payload_com=a.payload_com,
        seed=a.truth_seed, translation_mm=a.translation_mm, rotation_deg=a.rotation_deg), indent=1))
    (a.output/'metadata.json').write_text(json.dumps(dict(
        plan=str(a.plan), views=len(q_plan), board=plan['board'], image_size=[int(w_), int(h_)],
        gripper_opening_m=.02,
        physics_dt=1/240, settle_steps=a.settle_steps, rt_subframes=a.rt_subframes,
        renderer='RaytracedLighting, anti_aliasing=0', usd=str(USD), usd_urdf_origin_max_abs_diff=origin_check,
        encoder='drive target (motor side)', torque='get_measured_joint_efforts', hand_mass_model='URDF + payload',
        link_inertia='URDF diagonal (asset values ~1e-6 kg m^2 make stiff drives ill-conditioned)',
        notes='calibration reads samples.npz + images only; truth_* files are for evaluation'), indent=1))
    print('DONE', a.output, flush=True)


try:
    main()
finally:
    app.close()
