import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from panda_calib.board import Checkerboard
from panda_calib.camera import project, unpack_intrinsics
from panda_calib.export import write_urdf
from panda_calib.ik import compensate_compliance, solve_ik
from panda_calib.kinematics import PandaModel, invert, pose_error, transform
from panda_calib import synthetic as S

Q = np.array([[0., -.3, 0., -2.2, 0., 2., .785],
              [.4, .2, -.5, -1.6, .7, 1.2, -.3]])


class KinematicsTest(unittest.TestCase):
    model = PandaModel.from_urdf()

    def test_urdf_chain_matches_franka_dh_table(self):
        # Craig (modified) DH parameters from the Franka documentation, flange frame.
        a = [0., 0., 0., .0825, -.0825, 0., .088, 0.]
        d = [.333, 0., .316, 0., .384, 0., 0., .107]
        alpha = [0., -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0.]
        hand = transform([0., 0., -np.pi/4])  # panda_hand_joint
        for q in Q:
            T = np.eye(4)
            for i, theta in enumerate(np.r_[q, 0.]):
                T = T@transform([alpha[i], 0., 0.], [a[i], 0., 0.])@transform([0., 0., theta], [0., 0., d[i]])
            np.testing.assert_allclose(self.model.fk(q[None])[0], T@hand, atol=1e-9)  # URDF stores pi/2 to 11 digits

    def test_holding_torque_is_potential_gradient(self):
        geo = np.random.default_rng(0).normal(0, 1e-3, (8, 6))

        def potential(q):
            _, L = self.model.link_frames(q, geo)
            c = np.einsum('nkij,kj->nki', L[..., :3, :3], self.model.coms)+L[..., :3, 3]
            return -(self.model.masses*(c@self.model.gravity)).sum(1)

        h = 1e-6
        numeric = np.stack([(potential(Q+h*e)-potential(Q-h*e))/(2*h) for e in np.eye(7)], axis=1)
        np.testing.assert_allclose(self.model.holding_torque(Q, geo), numeric, atol=1e-6)

    def test_jacobian_matches_finite_differences(self):
        tool = transform(translation=[0., 0., .1034])
        J = self.model.jacobian(Q, None, tool)
        h = 1e-7
        for i in range(7):
            dq = np.zeros(7); dq[i] = h
            Tp, Tm = self.model.fk(Q+dq)@tool, self.model.fk(Q-dq)@tool
            np.testing.assert_allclose(J[:, :3, i], (Tp[:, :3, 3]-Tm[:, :3, 3])/(2*h), atol=1e-6)

    def test_ik_reaches_target_on_perturbed_model(self):
        geo = np.random.default_rng(1).normal(0, 1e-3, (8, 6))
        target = self.model.fk(Q[1:], geo)[0]
        q, ok, _ = solve_ik(self.model, target, Q[1]+.1, geo)
        self.assertTrue(ok)
        t, r = pose_error(self.model.fk(q[None], geo)[0], target)
        self.assertLess(t, 1e-6); self.assertLess(r, 1e-5)

    def test_compliance_compensation_inverts_settling(self):
        truth = S.random_truth(np.random.default_rng(2))
        q_cmd = compensate_compliance(self.model, Q, truth['compliance'], truth['geo'])
        q_link, _ = S.settle(self.model, q_cmd, truth)
        np.testing.assert_allclose(q_link, Q, atol=1e-9)

    def test_exported_urdf_reproduces_calibrated_fk(self):
        rng = np.random.default_rng(3)
        geo = rng.normal(0, 1e-3, (8, 6))
        T_hand_cam = transform(rng.normal(0, .1, 3), [.06, 0., .05])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cal.urdf'
            write_urdf(geo, T_hand_cam, path)
            exported = PandaModel.from_urdf(path)
            text = path.read_text()
        np.testing.assert_allclose(exported.fk(Q), self.model.fk(Q, geo), atol=1e-8)
        self.assertIn('camera_optical_frame', text)


class CameraBoardTest(unittest.TestCase):
    def test_projection_matches_opencv(self):
        x = np.array([900., 910., 630., 470., -.1, .05, 1e-3, -2e-3, .01])
        pts = np.random.default_rng(0).uniform([-.2, -.2, .4], [.2, .2, .8], (50, 3))
        K, dist = unpack_intrinsics(x)
        ref, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
        np.testing.assert_allclose(project(pts, x), ref.reshape(-1, 2), atol=1e-6)

    def render(self, board, T_cam_board, intrinsics, size=(1280, 960), ppm=4000):
        """Ideal pinhole image of the board via a plane homography."""
        s = board.square
        n = int(round(s*ppm))
        # Texture spans board x in [-2s, (cols+1)s]: the squares plus one white margin square.
        tex = np.full(((board.rows+3)*n, (board.cols+3)*n), 255, np.uint8)
        for i in range(board.cols+1):
            for j in range(board.rows+1):
                if board.square_is_black(i, j):  # square (i, j) spans [(i-1)s, i s]
                    tex[(j+1)*n:(j+2)*n, (i+1)*n:(i+2)*n] = 0
        A = np.array([[s/n, 0, (.5/n-2)*s], [0, s/n, (.5/n-2)*s], [0, 0, 1]])  # pixel centre -> board metres
        K, _ = unpack_intrinsics(intrinsics)
        H = K@np.c_[T_cam_board[:3, :2], T_cam_board[:3, 3]]@A
        return cv2.warpPerspective(tex, H, size, flags=cv2.INTER_AREA, borderValue=128)

    def test_detection_order_is_unique_for_any_in_plane_rotation(self):
        board = Checkerboard()
        intr = S.NOMINAL_INTRINSICS
        centre = board.object_points.mean(0)
        for roll in np.deg2rad([0., 50., 120., 180., 250., 320.]):
            T = (transform([np.pi, 0., 0.], [.02, -.01, .55])@transform([.3, -.2, 0.])@transform([0., 0., roll])
                 @transform(translation=-centre))
            image = self.render(board, T, intr)
            corners = board.detect(image)
            self.assertIsNotNone(corners, f'roll {roll}')
            pts = board.object_points@T[:3, :3].T+T[:3, 3]
            err = np.linalg.norm(corners-project(pts, intr), axis=1)
            self.assertLess(err.max(), .5, f'roll {np.degrees(roll)}')


if __name__ == '__main__':
    unittest.main()
