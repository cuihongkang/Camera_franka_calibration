"""Calibrate from a dataset directory (simulated or real robot).

Dataset layout:
  samples.npz     q (N,7) encoder angles [rad], tau (N,7) measured joint torques [N m],
                  image_size (2,) = (width, height)
  images/NNNNNN.png
  metadata.json   {"board": {"type": "charuco", "cols": 12, "rows": 9, "square": 0.03,
                             "marker": 0.022, "dictionary": "DICT_5X5_100"}, ...}
                  or {"board": {"type": "checkerboard", "cols": 11, "rows": 8, "square": 0.03}}

Gripper occlusion: unless ``--mask`` is given, the static region (the gripper, rigid in
the wrist camera) is estimated from the images and saved as gripper_mask.png in the
output directory.  Corners near it are discarded before calibration.

Every ``--test-every``-th view is held out for evaluation.  Outputs per mode:
results/<name>/<mode>/calibration.json and panda_calibrated.urdf.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from panda_calib.board import board_from_meta
from panda_calib.calibrate import MODES, Samples, calibrate, initialise, reprojection_rms
from panda_calib.export import save_result
from panda_calib.kinematics import PandaModel
from panda_calib.occlusion import estimate_static_mask, load_mask, save_mask

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset', type=Path)
p.add_argument('--output', type=Path, default=None)
p.add_argument('--modes', nargs='+', default=list(MODES), choices=list(MODES))
p.add_argument('--test-every', type=int, default=4, help='hold out every k-th view (0: use all for training)')
p.add_argument('--payload-kg', type=float, default=0., help='only used for compliance compensation in IK')
p.add_argument('--fix-intrinsics', action='store_true')
p.add_argument('--mask', default='auto', help="gripper mask PNG (white = gripper), 'auto' or 'none'")
a = p.parse_args()

meta = json.loads((a.dataset/'metadata.json').read_text())
board = board_from_meta(meta['board'])
data = np.load(a.dataset/'samples.npz')
out = a.output or Path('results')/a.dataset.name
out.mkdir(parents=True, exist_ok=True)
image = lambda n: np.asarray(Image.open(a.dataset/'images'/f'{n:06d}.png'))
if a.mask == 'none':
    mask = None
elif a.mask == 'auto':
    step = max(1, len(data['q'])//60)
    mask = estimate_static_mask([image(n) for n in range(0, len(data['q']), step)])
    save_mask(mask, out/'gripper_mask.png')
    print(f'estimated gripper mask: {mask.mean()*100:.1f}% of the image -> {out/"gripper_mask.png"}')
else:
    mask = load_mask(a.mask)
cache = out/'corners.npz'
mask_hash = hashlib.sha1(mask.tobytes() if mask is not None else b'').hexdigest()
if cache.exists() and str(np.load(cache)['mask_hash']) == mask_hash:
    corners, found = np.load(cache)['corners'], np.load(cache)['found']
else:
    corners = np.full((len(data['q']), len(board.object_points), 2), np.nan)
    found = np.zeros(len(data['q']), bool)
    for n in range(len(data['q'])):
        c = board.detect(image(n), mask)
        if c is not None:
            corners[n], found[n] = c, True
    np.savez(cache, corners=corners, found=found, mask_hash=mask_hash)
n_corners = np.isfinite(corners[found, :, 0]).sum(1)
print(f'board detected in {found.sum()}/{len(found)} views; corners per view min {n_corners.min()} '
      f'median {int(np.median(n_corners))} of {len(board.object_points)}')

index = np.flatnonzero(found)
test = index[::a.test_every] if a.test_every else index[:0]
train = np.setdiff1d(index, test)
samples = Samples(data['q'], data['tau'], corners, tuple(int(v) for v in data['image_size']))
train_s, test_s = samples.subset(train), samples.subset(test)
model = PandaModel.from_urdf()
init = initialise(model, board, train_s)
if a.fix_intrinsics:
    print('intrinsics fixed at the cv2.calibrateCamera estimate')
summary = dict(dataset=str(a.dataset), train=train.tolist(), test=test.tolist(), modes={})
for mode in a.modes:
    r = calibrate(model, board, train_s, mode, init=init, estimate_intrinsics=not a.fix_intrinsics)
    save_result(r, out/mode)
    test_rms = reprojection_rms(r, test_s) if len(test) else float('nan')
    summary['modes'][mode] = dict(train_rms_px=r['rms_px'], test_rms_px=test_rms, n_free=len(r['free']),
                                  condition=r['condition'])
    print(f'{mode:9s} train {r["rms_px"]:.3f} px  test {test_rms:.3f} px  free {len(r["free"])}  '
          f'cond {r["condition"]:.1e}')
    if 'compliance' in MODES[mode]:
        c = r['params']['compliance']
        print('  stiffness N m/rad:', {f'J{i+1}': (round(1/v) if v > 0 else 'n/a')
                                        for i, v in enumerate(c) if f'c{i+1}' in r['free']})
(out/'summary.json').write_text(json.dumps(summary, indent=1))
