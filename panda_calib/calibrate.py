"""Joint estimation of Panda frame errors, joint compliance, hand-eye and intrinsics.

Observations are board-corner detections from an eye-in-hand camera at static
configurations, with encoder angles and measured joint torques.  Corners that were
not observed (occluded by the gripper, outside the image) are NaN and contribute no
residual.  Residuals are corner reprojection errors in pixels.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import INTRINSIC_NAMES, pack_intrinsics, project
from .kinematics import N_FRAMES, invert, link_side_angles, transform

MODES = {
    # nominal kinematics: only board pose, hand-eye and intrinsics (classic hand-eye refinement)
    'handeye': ('board', 'hand_eye', 'intrinsics'),
    'geometry': ('board', 'hand_eye', 'intrinsics', 'geo'),
    'full': ('board', 'hand_eye', 'intrinsics', 'geo', 'compliance'),
}
AXES = 'xyz'


@dataclass
class Samples:
    q: np.ndarray        # (N,7) encoder readings, rad
    tau: np.ndarray      # (N,7) measured joint torques, N m
    corners: np.ndarray  # (N,K,2) detected corners in board order, px; NaN = not observed
    image_size: tuple    # (width, height)

    def subset(self, index):
        return Samples(self.q[index], self.tau[index], self.corners[index], self.image_size)

    @property
    def valid(self):
        return np.isfinite(self.corners).all(axis=-1)


@dataclass
class Layout:
    names: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    scales: list = field(default_factory=list)

    def add(self, group, names, scales):
        self.names += names; self.groups += [group]*len(names); self.scales += list(scales)

    def index(self, group):
        return np.array([i for i, g in enumerate(self.groups) if g == group])


def make_layout():
    L = Layout()
    for group in ['board', 'hand_eye']:
        L.add(group, [f'{group}.r{a}' for a in AXES]+[f'{group}.t{a}' for a in AXES], [1e-2]*3+[1e-2]*3)
    L.add('intrinsics', [f'K.{n}' for n in INTRINSIC_NAMES], [1., 1., 1., 1., 1e-2, 1e-2, 1e-3, 1e-3, 1e-2])
    L.add('compliance', [f'c{i}' for i in range(1, 8)], [1e-4]*7)
    frames = [f'J{i}' for i in range(1, 8)]+['flange']
    for f in frames:
        L.add('geo', [f'{f}.t{a}' for a in AXES]+[f'{f}.r{a}' for a in AXES], [1e-3]*6)
    return L


def priority(layout):
    """Order used when choosing a non-redundant parameter subset: keep earlier ones."""
    first = [i for g in ['board', 'hand_eye', 'intrinsics', 'compliance'] for i in layout.index(g)]
    geo = []
    for f in [f'J{i}' for i in range(2, 8)]+['J1', 'flange']:
        for suffix in ['rz', 'rx', 'ry', 'tx', 'ty', 'tz']:
            geo.append(layout.names.index(f'{f}.{suffix}'))
    return first+geo


class Problem:
    def __init__(self, model, board, samples, T_base_board0, T_hand_cam0):
        self.model, self.board, self.samples = model, board, samples
        self.T_base_board0, self.T_hand_cam0 = T_base_board0, T_hand_cam0
        self.layout = make_layout()
        self.scales = np.array(self.layout.scales)
        self.points = board.object_points

    def unpack(self, x):
        L = self.layout
        b, h = x[L.index('board')], x[L.index('hand_eye')]
        return dict(T_base_board=self.T_base_board0@transform(b[:3], b[3:]),
                    T_hand_cam=self.T_hand_cam0@transform(h[:3], h[3:]),
                    intrinsics=x[L.index('intrinsics')],
                    compliance=x[L.index('compliance')],
                    geo=x[L.index('geo')].reshape(N_FRAMES, 6))

    def camera_poses(self, p, samples=None):
        s = self.samples if samples is None else samples
        q_link = link_side_angles(s.q, s.tau, p['compliance'])
        return self.model.fk(q_link, p['geo'])@p['T_hand_cam']

    def predict(self, p, samples=None):
        T_cam_board = invert(self.camera_poses(p, samples))@p['T_base_board']
        pts = np.einsum('nij,kj->nki', T_cam_board[:, :3, :3], self.points)+T_cam_board[:, None, :3, 3]
        return project(pts, p['intrinsics'])

    def residual(self, x, samples=None):
        s = self.samples if samples is None else samples
        return (self.predict(self.unpack(x), s)-s.corners)[s.valid].ravel()


def initialise(model, board, samples, T_hand_cam_prior=None):
    """Intrinsics from cv2.calibrateCamera, hand-eye from Park's method on nominal FK."""
    valid = samples.valid
    obj = [board.object_points[v].astype(np.float32) for v in valid]
    img = [c[v].astype(np.float32).reshape(-1, 1, 2) for c, v in zip(samples.corners, valid)]
    # Partial (non-coplanar-grid) views need an intrinsic guess; a centred pinhole is enough.
    w, h = samples.image_size
    K0 = np.array([[w, 0., w/2], [0., w, h/2], [0., 0., 1.]])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, tuple(samples.image_size), K0, None,
                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    T_cam_board = transform(np.array(rvecs).reshape(-1, 3), np.array(tvecs).reshape(-1, 3))
    T_base_hand = model.fk(samples.q)
    if T_hand_cam_prior is None:
        R, t = cv2.calibrateHandEye(T_base_hand[:, :3, :3], T_base_hand[:, :3, 3], T_cam_board[:, :3, :3],
                                    T_cam_board[:, :3, 3], method=cv2.CALIB_HAND_EYE_PARK)
        T_hand_cam = np.eye(4); T_hand_cam[:3, :3] = R; T_hand_cam[:3, 3] = t.ravel()
    else:
        T_hand_cam = np.asarray(T_hand_cam_prior, float)
    T = T_base_hand@T_hand_cam@T_cam_board
    T_base_board = np.eye(4)
    T_base_board[:3, :3] = Rotation.from_matrix(T[:, :3, :3]).mean().as_matrix()
    T_base_board[:3, 3] = np.median(T[:, :3, 3], axis=0)
    return dict(intrinsics=pack_intrinsics(K, dist), T_hand_cam=T_hand_cam, T_base_board=T_base_board,
                camera_rms_px=float(rms))


def numerical_jacobian(fun, x, scales, step=1e-4):
    """Central differences with steps proportional to the parameter scales."""
    f0 = fun(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = step*scales[i]
        e = np.zeros_like(x); e[i] = h
        J[:, i] = (fun(x+e)-fun(x-e))/(2*h)
    return J


def select_identifiable(J_scaled, order, candidates, tol=1e-4, min_sensitivity=.005):
    """Greedy Gram-Schmidt in priority order; drop columns inside the span of kept ones.

    ``min_sensitivity`` (pixels, RMS per corner coordinate for a change of one parameter
    scale) also drops parameters the data barely excites, e.g. the compliance of a joint
    whose gravity torque is always ~0.  Returns kept indices and, for each candidate, the
    relative norm of the part of its column not explained by the previously kept columns
    (0 = exactly redundant).
    """
    basis, kept, ratio = [], [], {}
    rms = np.linalg.norm(J_scaled, axis=0)/np.sqrt(len(J_scaled))
    for i in order:
        if i not in candidates:
            continue
        if rms[i] < min_sensitivity:
            ratio[i] = 0.
            continue
        v = J_scaled[:, i].copy()
        norm = np.linalg.norm(v)
        for _ in range(2):  # re-orthogonalise for numerical stability
            for b in basis:
                v -= (b@v)*b
        r = np.linalg.norm(v)/norm if norm > 0 else 0.
        ratio[i] = r
        if r > tol:
            basis.append(v/np.linalg.norm(v)); kept.append(i)
    return sorted(kept), ratio


def calibrate(model, board, samples, mode='full', init=None, estimate_intrinsics=True, tol=1e-4, min_sensitivity=.005,
              loss='huber', f_scale=1., max_nfev=200, verbose=0):
    init = initialise(model, board, samples) if init is None else init
    problem = Problem(model, board, samples, init['T_base_board'], init['T_hand_cam'])
    L = problem.layout
    x0 = np.zeros(len(L.names))
    x0[L.index('intrinsics')] = init['intrinsics']
    groups = list(MODES[mode])
    if not estimate_intrinsics:
        groups.remove('intrinsics')
    candidates = {i for g in groups for i in L.index(g)}
    J = numerical_jacobian(problem.residual, x0, problem.scales)*problem.scales
    free, ratio = select_identifiable(J, priority(L), candidates, tol, min_sensitivity)
    free = np.array(free)

    def fun(z):
        x = x0.copy(); x[free] = z
        return problem.residual(x)

    result = least_squares(fun, x0[free], x_scale=problem.scales[free], loss=loss, f_scale=f_scale,
                           method='trf', max_nfev=max_nfev, verbose=verbose)
    x = x0.copy(); x[free] = result.x
    r = problem.residual(x)
    dof = max(r.size-free.size, 1)
    sigma2 = float(r@r)/dof
    Js = numerical_jacobian(fun, result.x, problem.scales[free])
    try:
        std = np.sqrt(np.diag(np.linalg.inv(Js.T@Js))*sigma2)
    except np.linalg.LinAlgError:
        std = np.full(free.size, np.nan)
    params = problem.unpack(x)
    d = np.zeros(samples.corners.shape); d[samples.valid] = r.reshape(-1, 2)
    per_image = np.sqrt((d**2).sum(-1).sum(-1)/samples.valid.sum(-1))
    return dict(mode=mode, params=params, x=x, free=[L.names[i] for i in free],
                std={L.names[i]: float(s) for i, s in zip(free, std)},
                fixed_redundant=[L.names[i] for i in sorted(candidates-set(free.tolist()))],
                independence={L.names[i]: float(v) for i, v in ratio.items()},
                rms_px=float(np.sqrt(np.mean(r**2)*2)), per_image_rms_px=per_image.tolist(),
                condition=float(np.linalg.cond(Js/np.linalg.norm(Js, axis=0))),
                init=init, optimizer=dict(status=int(result.status), nfev=int(result.nfev), message=result.message),
                problem=problem)


def reprojection_rms(result, samples):
    problem = result['problem']
    d = (problem.predict(result['params'], samples)-samples.corners)[samples.valid]
    return float(np.sqrt((d**2).sum(-1).mean()))
