"""End-to-end recovery on synthetic corner observations with known truth."""
import unittest

import numpy as np

from panda_calib.board import Checkerboard
from panda_calib.calibrate import Samples, calibrate, reprojection_rms
from panda_calib.kinematics import PandaModel, invert, pose_error
from panda_calib.poses import sample_configurations
from panda_calib import synthetic as S


class SyntheticRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.model, cls.board = PandaModel.from_urdf(), Checkerboard()
        q, _ = sample_configurations(cls.model, cls.board, S.BOARD_POSE, S.NOMINAL_MOUNT, S.NOMINAL_INTRINSICS,
                                     S.IMAGE_SIZE, 100, rng)
        cls.truth = S.random_truth(rng)
        uv, tau, cls.T_cam, _ = S.observe(cls.model, cls.board, q, cls.truth, rng, pixel_noise=.15)
        samples = Samples(q, tau, uv, S.IMAGE_SIZE)
        cls.train, cls.test = samples.subset(slice(0, 70)), samples.subset(slice(70, None))
        cls.results = {mode: calibrate(cls.model, cls.board, cls.train, mode) for mode in ['handeye', 'full']}

    def camera_error_mm(self, result):
        p, problem = result['params'], result['problem']
        est = invert(invert(problem.camera_poses(p, self.test))@p['T_base_board'])
        true = invert(invert(self.T_cam[70:])@self.truth['T_base_board'])
        return pose_error(est, true)[0]*1e3

    def test_full_model_reaches_noise_floor_on_held_out_views(self):
        full = self.results['full']
        self.assertLess(reprojection_rms(full, self.test), .3)
        self.assertLess(self.camera_error_mm(full).max(), .3)

    def test_nominal_kinematics_cannot_explain_the_data(self):
        self.assertGreater(reprojection_rms(self.results['handeye'], self.test), 2.)
        self.assertGreater(self.camera_error_mm(self.results['handeye']).mean(), 1.)

    def test_compliance_of_loaded_joints(self):
        est = self.results['full']['params']['compliance']
        for j in [1, 3]:  # joints 2 and 4 carry most of the gravity load
            self.assertAlmostEqual(est[j]/self.truth['compliance'][j], 1., delta=.05)

    def test_redundant_parameters_are_fixed(self):
        fixed = set(self.results['full']['fixed_redundant'])
        # Base and flange frames are absorbed by the board pose and hand-eye transform.
        self.assertTrue({f'J1.{a}' for a in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']} <= fixed)
        self.assertIn('c7', fixed)
        n_geo = sum(name.startswith(('J', 'flange')) for name in self.results['full']['free'])
        self.assertEqual(n_geo+12, 4*7+6)  # complete, minimal: 4R + 6 with board and hand-eye


class PartialViewTest(unittest.TestCase):
    """Views with a third of the corners missing (gripper occlusion) must give the same accuracy."""

    def test_recovery_with_missing_corners(self):
        rng = np.random.default_rng(4)
        model, board = PandaModel.from_urdf(), Checkerboard()
        q, _ = sample_configurations(model, board, S.BOARD_POSE, S.NOMINAL_MOUNT, S.NOMINAL_INTRINSICS,
                                     S.IMAGE_SIZE, 80, rng)
        truth = S.random_truth(rng)
        uv, tau, T_cam, _ = S.observe(model, board, q, truth, rng, pixel_noise=.15)
        # Drop every corner in the left third of the image, as a wrist gripper would.
        uv[uv[..., 0] < S.IMAGE_SIZE[0]/3] = np.nan
        samples = Samples(q, tau, uv, S.IMAGE_SIZE)
        self.assertLess(samples.valid.mean(), .95)
        train, test = samples.subset(slice(0, 60)), samples.subset(slice(60, None))
        r = calibrate(model, board, train, 'full')
        self.assertLess(reprojection_rms(r, test), .3)
        p = r['params']
        est = invert(invert(r['problem'].camera_poses(p, test))@p['T_base_board'])
        true = invert(invert(T_cam[60:])@truth['T_base_board'])
        self.assertLess(pose_error(est, true)[0].max()*1e3, .3)


if __name__ == '__main__':
    unittest.main()
