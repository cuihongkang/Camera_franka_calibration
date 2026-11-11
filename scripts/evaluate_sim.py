"""Compare calibrated models with the simulation ground truth on held-out views.

Metrics per mode:
  camera   predicted base->camera pose vs. renderer camera pose
  tcp      predicted Franka fingertip TCP (hand z + 0.1034 m) vs. physics; includes the
           part of the joint-7/flange error that the hand-eye transform absorbs and the
           camera cannot see (README limitations)
  compliance  estimate vs. injected value for the joints that were estimated
"""
import argparse
import json
from pathlib import Path

import numpy as np

from panda_calib.export import load_result
from panda_calib.kinematics import PandaModel, invert, link_side_angles, pose_error, transform

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset', type=Path)
p.add_argument('--results', type=Path, default=None)
p.add_argument('--tcp', type=float, default=.1034)
a = p.parse_args()
results = a.results or Path('results')/a.dataset.name

truth = json.loads((a.dataset/'truth.json').read_text())
gt = np.load(a.dataset/'truth.npz')
data = np.load(a.dataset/'samples.npz')
summary = json.loads((results/'summary.json').read_text())
test = np.array(summary['test'])
model = PandaModel.from_urdf()
T_base_board = np.array(truth['T_base_board'])
T_hand_tcp = transform(translation=[0., 0., a.tcp])
true_cam = gt['T_world_cam'][test]
true_hand = gt['T_world_hand'][test]

report = {}
for mode in summary['modes']:
    cal = load_result(results/mode/'calibration.json')
    q_link = link_side_angles(data['q'][test], data['tau'][test], cal['compliance'])
    hand = model.fk(q_link, cal['geo'])
    # The board pose fixes the base gauge: express predictions in the true world through it.
    to_world = T_base_board@invert(cal['T_base_board'])
    cam = to_world@hand@cal['T_hand_cam']
    e_cam = pose_error(cam, true_cam)
    e_hand = pose_error(to_world@hand, true_hand)
    e_tcp = pose_error(to_world@hand@T_hand_tcp, true_hand@T_hand_tcp)
    est = dict(zip([f'c{i}' for i in range(1, 8)], cal['compliance']))
    free = json.loads((results/mode/'calibration.json').read_text())['estimated']
    report[mode] = dict(
        camera_mm=dict(mean=float(e_cam[0].mean()*1e3), max=float(e_cam[0].max()*1e3)),
        camera_deg=dict(mean=float(np.degrees(e_cam[1]).mean()), max=float(np.degrees(e_cam[1]).max())),
        tcp_mm=dict(mean=float(e_tcp[0].mean()*1e3), max=float(e_tcp[0].max()*1e3)),
        hand_deg=dict(mean=float(np.degrees(e_hand[1]).mean()), max=float(np.degrees(e_hand[1]).max())),
        stiffness_Nm_per_rad={k: dict(estimated=float(1/v), true=float(1/truth['compliance'][int(k[1])-1]))
                              for k, v in est.items() if k in free and v > 0},
        test_rms_px=summary['modes'][mode]['test_rms_px'])
    print(f"{mode:9s} camera {report[mode]['camera_mm']['mean']:.3f}/{report[mode]['camera_mm']['max']:.3f} mm "
          f"{report[mode]['camera_deg']['mean']:.4f} deg   tcp {report[mode]['tcp_mm']['mean']:.3f}/"
          f"{report[mode]['tcp_mm']['max']:.3f} mm   test {report[mode]['test_rms_px']:.3f} px")
    for k, v in report[mode]['stiffness_Nm_per_rad'].items():
        print(f'    K{k[1]}: {v["estimated"]:.0f} (true {v["true"]:.0f})')
(results/'evaluation.json').write_text(json.dumps(report, indent=1))
