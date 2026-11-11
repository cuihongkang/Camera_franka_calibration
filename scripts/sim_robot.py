"""Isaac Sim helpers: a Franka with injected joint-frame errors and compliant joint drives.

Must be imported after SimulationApp has started.  The URDF-style error E_i
(see panda_calib.kinematics) is written into the parent-side joint frame:
``O_i E_i Rz(q) = F0' Rx(q) F1^-1`` gives ``F0' = F0 F1^-1 E_i F1``.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from panda_calib.kinematics import ARM_JOINTS, transform

ARM_PATHS = [f'panda_link{i}/{name}' for i, name in enumerate(ARM_JOINTS)]
FLANGE_PATH = 'panda_link7/panda_hand_joint'


def frame(prim, side):
    pos = prim.GetAttribute(f'physics:localPos{side}').Get()
    rot = prim.GetAttribute(f'physics:localRot{side}').Get()
    T = np.eye(4)
    T[:3, :3] = np.array(Gf.Matrix3d(Gf.Quatd(rot.GetReal(), Gf.Vec3d(rot.GetImaginary())))).T
    T[:3, 3] = np.array(pos)
    return T


def set_frame(prim, side, T):
    q = Gf.Matrix4d(T.T.tolist()).ExtractRotationQuat()
    prim.GetAttribute(f'physics:localPos{side}').Set(Gf.Vec3f(*map(float, T[:3, 3])))
    prim.GetAttribute(f'physics:localRot{side}').Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*map(float, q.GetImaginary()))))


def joint_origins(stage, root):
    """Nominal URDF-equivalent origins O = F0 F1^-1 read back from the USD joints."""
    out = []
    for path in ARM_PATHS+[FLANGE_PATH]:
        prim = stage.GetPrimAtPath(f'{root}/{path}')
        out.append(frame(prim, 0)@np.linalg.inv(frame(prim, 1)))
    return np.stack(out)


def inject_errors(stage, root, geo):
    """Write per-joint frame errors (8,6) [t, rotvec] into the joint prims."""
    for i, path in enumerate(ARM_PATHS+[FLANGE_PATH]):
        prim = stage.GetPrimAtPath(f'{root}/{path}')
        if not prim.IsValid() or prim.IsInstanceProxy():
            raise RuntimeError(f'cannot edit joint {path}')
        F0, F1 = frame(prim, 0), frame(prim, 1)
        set_frame(prim, 0, F0@np.linalg.inv(F1)@transform(geo[i, 3:], geo[i, :3])@F1)


def set_joint_drives(stage, root, stiffness, damping, max_force=None, iterations=64):
    """Angular drive gains given in SI (N m/rad, N m s/rad); USD expects per-degree units."""
    for i, path in enumerate(ARM_PATHS):
        prim = stage.GetPrimAtPath(f'{root}/{path}')
        drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
        drive.GetStiffnessAttr().Set(float(stiffness[i]*np.pi/180))
        drive.GetDampingAttr().Set(float(damping[i]*np.pi/180))
        if max_force is not None:
            drive.GetMaxForceAttr().Set(float(max_force[i]))
        for name in ['physxJoint:jointFriction', 'physxJoint:armature']:
            attr = prim.GetAttribute(name)
            if attr.IsValid():
                attr.Set(0.)
    root_prim = stage.GetPrimAtPath(root)
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            api.CreateSolverPositionIterationCountAttr().Set(iterations)
            api.CreateSolverVelocityIterationCountAttr().Set(4)
            # Sleeping freezes the arm before it reaches static equilibrium under gravity.
            api.CreateSleepThresholdAttr().Set(0.)
            api.CreateStabilizationThresholdAttr().Set(0.)


def material(stage, path, rgb):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0., 0., 0.))
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def quad_mesh(stage, path, rects, z, mat):
    """Mesh of axis-aligned rectangles [(x0, y0, x1, y1)] at height z (local frame)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = [], [], []
    for x0, y0, x1, y1 in rects:
        base = len(pts)
        pts += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        counts.append(4); idx += [base, base+1, base+2, base+3]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)]*len(pts)); mesh.SetNormalsInterpolation('vertex')
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def build_board(stage, board, T_world_board):
    xf = UsdGeom.Xform.Define(stage, '/World/Board')
    xf.AddTransformOp().Set(Gf.Matrix4d(T_world_board.T.tolist()))
    s = board.square
    white, black = material(stage, '/World/Looks/White', (.85, .85, .85)), material(stage, '/World/Looks/Black', (.02, .02, .02))
    # White substrate with one margin square beyond the outer squares.
    quad_mesh(stage, '/World/Board/Substrate', [(-2*s, -2*s, (board.cols+1)*s, (board.rows+1)*s)], -5e-4, white)
    rects = [((i-1)*s, (j-1)*s, i*s, j*s) for i in range(board.cols+1) for j in range(board.rows+1)
             if board.square_is_black(i, j)]
    quad_mesh(stage, '/World/Board/Black', rects, 0., black)
    floor = material(stage, '/World/Looks/Floor', (.35, .37, .4))
    quad_mesh(stage, '/World/Floor', [(-2., -2., 2., 2.)], -.01, floor)


def set_payload(stage, root, mass, com):
    """Add a point mass (camera, mount) to panda_hand."""
    prim = stage.GetPrimAtPath(f'{root}/panda_hand')
    api = UsdPhysics.MassAPI(prim)
    m0 = api.GetMassAttr().Get()
    c0 = np.array(api.GetCenterOfMassAttr().Get() or (0., 0., 0.))
    total = m0+mass
    api.GetMassAttr().Set(float(total))
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, (m0*c0+mass*np.asarray(com))/total)))


def set_urdf_inertia(stage, root, urdf):
    """Replace the asset's link inertias (~1e-6 kg m^2, far below the URDF) by the URDF values."""
    import xml.etree.ElementTree as ET
    for link in ET.parse(urdf).getroot().findall('link'):
        node = link.find('inertial')
        prim = stage.GetPrimAtPath(f"{root}/{link.get('name')}")
        if node is None or not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        I = node.find('inertia').attrib
        if any(abs(float(I[k])) > 0 for k in ['ixy', 'ixz', 'iyz']):
            raise ValueError(f"{link.get('name')}: non-diagonal URDF inertia not supported")
        api = UsdPhysics.MassAPI(prim)
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(float(I['ixx']), float(I['iyy']), float(I['izz'])))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1., 0., 0., 0.))
