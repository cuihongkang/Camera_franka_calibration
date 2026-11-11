"""Panda forward kinematics with per-joint frame errors and joint compliance.

Joint i (1..7) follows the URDF convention ``T = O_i @ E_i @ Rz(q_i)`` where
``O_i`` is the nominal URDF origin and ``E_i`` a small rigid error (translation
in metres, rotation vector in radians, both expressed in the nominal origin
frame).  Index 7 of the error array is the fixed flange (link7 -> panda_hand).

Joint compliance maps encoder readings to link-side angles:
``q_link = q_encoder - compliance * tau`` with ``tau`` the torque the joint
applies (Franka ``tau_J`` sign; it equals the gravity holding torque at rest).
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
N_FRAMES = 8  # 7 revolute joints + flange
GRAVITY = np.array([0., 0., -9.81])
DEFAULT_URDF = Path(__file__).resolve().parents[1]/'configs/panda_arm_hand.urdf'


def transform(rotvec=None, translation=None):
    """Homogeneous transform(s) from rotation vectors and translations."""
    rotvec = np.zeros(3) if rotvec is None else np.asarray(rotvec, float)
    translation = np.zeros(rotvec.shape) if translation is None else np.asarray(translation, float)
    shape = np.broadcast_shapes(rotvec.shape, translation.shape)[:-1]
    T = np.zeros(shape+(4, 4))
    T[..., :3, :3] = Rotation.from_rotvec(np.array(np.broadcast_to(rotvec, shape+(3,))).reshape(-1, 3)).as_matrix().reshape(shape+(3, 3))
    T[..., :3, 3] = translation
    T[..., 3, 3] = 1.
    return T


def to_rotvec_translation(T):
    T = np.asarray(T)
    return Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(T.shape[:-2]+(3,)), T[..., :3, 3]


def invert(T):
    T = np.asarray(T)
    out = np.zeros_like(T)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', R, T[..., :3, 3])
    out[..., 3, 3] = 1.
    return out


def rot_z(q):
    q = np.asarray(q, float)
    T = np.zeros(q.shape+(4, 4))
    c, s = np.cos(q), np.sin(q)
    T[..., 0, 0] = c; T[..., 0, 1] = -s; T[..., 1, 0] = s; T[..., 1, 1] = c
    T[..., 2, 2] = 1.; T[..., 3, 3] = 1.
    return T


def urdf_origin(node):
    origin = node.find('origin')
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()  # URDF fixed-axis roll, pitch, yaw
    T[:3, 3] = xyz
    return T


def pose_error(T_est, T_true):
    """Translation error (m) and rotation error (rad) between pose arrays."""
    D = invert(T_true) @ T_est
    angle = np.linalg.norm(Rotation.from_matrix(D[..., :3, :3].reshape(-1, 3, 3)).as_rotvec(), axis=-1).reshape(D.shape[:-2])
    return np.linalg.norm(T_est[..., :3, 3]-T_true[..., :3, 3], axis=-1), angle


class PandaModel:
    """Nominal Panda kinematic chain plus lumped link masses for gravity torques."""

    def __init__(self, origins, lower, upper, masses, coms, gravity=GRAVITY):
        self.origins = np.asarray(origins, float)    # (8,4,4); index 7 = link7 -> panda_hand
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.masses = np.asarray(masses, float)      # (8,) link1..link7, hand (incl. fingers)
        self.coms = np.asarray(coms, float)          # (8,3) in the respective link frame
        self.gravity = np.asarray(gravity, float)

    @classmethod
    def from_urdf(cls, path=DEFAULT_URDF, extra_hand_mass=0., extra_hand_com=(0., 0., 0.)):
        root = ET.parse(path).getroot()
        joint = {j.get('name'): j for j in root.findall('joint')}
        link = {l.get('name'): l for l in root.findall('link')}
        origins = [urdf_origin(joint[name]) for name in ARM_JOINTS]
        origins.append(urdf_origin(joint['panda_joint8'])@urdf_origin(joint['panda_hand_joint']))
        lower = [float(joint[n].find('limit').get('lower')) for n in ARM_JOINTS]
        upper = [float(joint[n].find('limit').get('upper')) for n in ARM_JOINTS]

        def inertial(name):
            node = link[name].find('inertial')
            if node is None:
                return 0., np.zeros(3)
            return float(node.find('mass').get('value')), urdf_origin(node)[:3, 3]

        masses, coms = [], []
        for i in range(1, 8):
            m, c = inertial(f'panda_link{i}')
            masses.append(m); coms.append(c)
        # Lump hand, both fingers and any mounted camera/payload into the hand frame.
        m_hand, c_hand = inertial('panda_hand')
        parts = [(m_hand, c_hand), (extra_hand_mass, np.asarray(extra_hand_com, float))]
        finger_offset = urdf_origin(joint['panda_finger_joint1'])[:3, 3]
        for finger in ['panda_leftfinger', 'panda_rightfinger']:
            m, c = inertial(finger)
            parts.append((m, c+finger_offset))
        total = sum(m for m, _ in parts)
        masses.append(total)
        coms.append(sum(m*c for m, c in parts)/max(total, 1e-12))
        return cls(np.stack(origins), lower, upper, masses, np.stack(coms))

    def link_frames(self, q_link, geo=None):
        """Return (joint_frames (N,7,4,4), link_frames (N,8,4,4)) in the base frame.

        joint_frames[:, i] is the frame of joint i before its rotation (z = joint axis);
        link_frames[:, i] is link i+1 (index 7 = panda_hand).
        """
        q_link = np.atleast_2d(q_link)
        n = len(q_link)
        local = self.origins if geo is None else self.origins@transform(geo[:, 3:], geo[:, :3])
        joint_frames = np.empty((n, 7, 4, 4))
        link_frames = np.empty((n, 8, 4, 4))
        T = np.broadcast_to(np.eye(4), (n, 4, 4))
        for i in range(7):
            T = T@local[i]
            joint_frames[:, i] = T
            T = T@rot_z(q_link[:, i])
            link_frames[:, i] = T
        link_frames[:, 7] = T@local[7]
        return joint_frames, link_frames

    def fk(self, q_link, geo=None):
        """Base -> panda_hand transforms for link-side joint angles."""
        return self.link_frames(q_link, geo)[1][:, 7]

    def holding_torque(self, q_link, geo=None):
        """Joint torque that balances gravity, same sign as Franka tau_J."""
        joint_frames, link_frames = self.link_frames(q_link, geo)
        coms = np.einsum('nkij,kj->nki', link_frames[..., :3, :3], self.coms)+link_frames[..., :3, 3]
        forces = self.masses[:, None]*self.gravity  # (8,3)
        tau = np.empty((len(coms), 7))
        for i in range(7):
            z = joint_frames[:, i, :3, 2]
            p = joint_frames[:, i, :3, 3]
            moment = np.cross(coms[:, i:]-p[:, None], forces[i:]).sum(axis=1)
            tau[:, i] = -np.einsum('ni,ni->n', z, moment)
        return tau

    def jacobian(self, q_link, geo=None, T_hand_tool=np.eye(4)):
        """Geometric Jacobian (N,6,7): linear velocity of the tool point, then angular velocity."""
        joint_frames, link_frames = self.link_frames(q_link, geo)
        tool = link_frames[:, 7]@T_hand_tool
        z = joint_frames[..., :3, 2]
        r = tool[:, None, :3, 3]-joint_frames[..., :3, 3]
        J = np.empty((len(tool), 6, 7))
        J[:, :3] = np.swapaxes(np.cross(z, r), 1, 2)
        J[:, 3:] = np.swapaxes(z, 1, 2)
        return J


def link_side_angles(q_encoder, tau, compliance):
    """Apply the joint compliance model to encoder readings."""
    return np.asarray(q_encoder)-np.asarray(compliance)*np.asarray(tau)
