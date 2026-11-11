"""Damped least-squares IK on a (calibrated) Panda model, with compliance pre-compensation."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_residual(T, T_target):
    dp = T_target[..., :3, 3]-T[..., :3, 3]
    dR = T_target[..., :3, :3]@np.swapaxes(T[..., :3, :3], -1, -2)
    return np.concatenate([dp, Rotation.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec().reshape(dp.shape)], axis=-1)


def _dls(model, T_target, q0, q_ref, geo, T_hand_tool, max_iter, damping, nullspace_gain, lower, upper, tol):
    q = np.array(q0, float)
    for _ in range(max_iter):
        T = model.fk(q[None], geo)[0]@T_hand_tool
        e = pose_residual(T, T_target)
        err = float(np.sqrt(e@e))
        if err < tol:
            break
        J = model.jacobian(q[None], geo, T_hand_tool)[0]
        JJt = J@J.T+damping**2*np.eye(6)
        dq = J.T@np.linalg.solve(JJt, e)
        if err > 1e-4:  # the damped projector leaks, so drop the secondary task near convergence
            dq += (np.eye(7)-J.T@np.linalg.solve(JJt, J))@(nullspace_gain*(q_ref-q))
        step = np.abs(dq).max()
        if step > .3:
            dq *= .3/step
        q = np.clip(q+dq, lower, upper)
    T = model.fk(q[None], geo)[0]@T_hand_tool
    e = pose_residual(T, T_target)
    return q, (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))


def solve_ik(model, T_target, q_ref, geo=None, T_hand_tool=np.eye(4), max_iter=300, damping=1e-3,
             nullspace_gain=.1, margin=1e-3, tol=1e-10, restarts=8, seed=0):
    """Link-side joint angles reaching ``T_target`` (base -> tool), staying near ``q_ref``.

    Starts at ``q_ref``; if that fails, retries from perturbed seeds and keeps the
    converged solution closest to ``q_ref``.  Returns (q, converged, residual) with
    residual = [position m, rotation rad] norms.
    """
    lower, upper = model.lower+margin, model.upper-margin
    q_ref = np.clip(np.asarray(q_ref, float), lower, upper)
    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(restarts+1):
        q0 = q_ref if attempt == 0 else np.clip(q_ref+rng.normal(0., .5, 7), lower, upper)
        q, res = _dls(model, T_target, q0, q_ref, geo, T_hand_tool, max_iter, damping, nullspace_gain,
                      lower, upper, tol)
        ok = res[0] < 1e-6 and res[1] < 1e-5
        dist = float(np.abs(q-q_ref).max())
        if best is None or (ok, -dist) > (best[1], -best[3]):
            best = (q, ok, res, dist)
        if ok and attempt == 0:
            break
    return best[0], bool(best[1]), best[2]


def compensate_compliance(model, q_link, compliance, geo=None, tau=None):
    """Encoder command whose loaded, link-side angle equals ``q_link``.

    ``tau`` defaults to the model gravity holding torque at ``q_link``; pass a measured
    torque instead when the link masses are not trusted (iterate once or twice).
    """
    q_link = np.asarray(q_link, float)
    if tau is None:
        tau = model.holding_torque(q_link[None], geo)[0] if q_link.ndim == 1 else model.holding_torque(q_link, geo)
    return q_link+np.asarray(compliance)*tau
