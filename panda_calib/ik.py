"""Damped least-squares IK on a (calibrated) Panda model, with compliance pre-compensation."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_residual(T, T_target):
    dp = T_target[..., :3, 3]-T[..., :3, 3]
    dR = T_target[..., :3, :3]@np.swapaxes(T[..., :3, :3], -1, -2)
    return np.concatenate([dp, Rotation.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec().reshape(dp.shape)], axis=-1)


def solve_ik(model, T_target, q_ref, geo=None, T_hand_tool=np.eye(4), max_iter=300, damping=1e-3,
             nullspace_gain=.2, margin=1e-3, tol=1e-9):
    """Link-side joint angles reaching ``T_target`` (base -> tool), staying near ``q_ref``.

    Returns (q, converged, residual) with residual = [position m, rotation rad] norms.
    """
    q = np.array(q_ref, float)
    lower, upper = model.lower+margin, model.upper-margin
    for _ in range(max_iter):
        T = model.fk(q[None], geo)[0]@T_hand_tool
        e = pose_residual(T, T_target)
        if e@e < tol**2:
            break
        J = model.jacobian(q[None], geo, T_hand_tool)[0]
        JJt = J@J.T+damping**2*np.eye(6)
        dq = J.T@np.linalg.solve(JJt, e)
        N = np.eye(7)-J.T@np.linalg.solve(JJt, J)
        dq += N@(nullspace_gain*(q_ref-q))
        q = np.clip(q+dq, lower, upper)
    T = model.fk(q[None], geo)[0]@T_hand_tool
    e = pose_residual(T, T_target)
    res = (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
    return q, bool(res[0] < 1e-6 and res[1] < 1e-5), res


def compensate_compliance(model, q_link, compliance, geo=None, tau=None):
    """Encoder command whose loaded, link-side angle equals ``q_link``.

    ``tau`` defaults to the model gravity holding torque at ``q_link``; pass a measured
    torque instead when the link masses are not trusted (iterate once or twice).
    """
    q_link = np.asarray(q_link, float)
    if tau is None:
        tau = model.holding_torque(q_link[None], geo)[0] if q_link.ndim == 1 else model.holding_torque(q_link, geo)
    return q_link+np.asarray(compliance)*tau
