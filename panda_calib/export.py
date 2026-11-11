"""Save calibration results as JSON and as a corrected URDF."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import INTRINSIC_NAMES
from .kinematics import ARM_JOINTS, DEFAULT_URDF, transform, urdf_origin


def _set_origin(node, T):
    origin = node.find('origin')
    if origin is None:
        origin = ET.SubElement(node, 'origin')
    origin.set('xyz', ' '.join(f'{v:.9f}' for v in T[:3, 3]))
    origin.set('rpy', ' '.join(f'{v:.9f}' for v in Rotation.from_matrix(T[:3, :3]).as_euler('xyz')))


def write_urdf(geo, T_hand_cam, output, urdf=DEFAULT_URDF):
    """Corrected URDF: joint origins O_i E_i, flange error in panda_hand_joint, camera frame."""
    tree = ET.parse(urdf)
    root = tree.getroot()
    joints = {j.get('name'): j for j in root.findall('joint')}
    for i, name in enumerate(ARM_JOINTS):
        _set_origin(joints[name], urdf_origin(joints[name])@transform(geo[i, 3:], geo[i, :3]))
    hand = joints['panda_hand_joint']
    _set_origin(hand, urdf_origin(hand)@transform(geo[7, 3:], geo[7, :3]))
    ET.SubElement(root, 'link', name='camera_optical_frame')
    cam = ET.SubElement(root, 'joint', name='camera_optical_joint', type='fixed')
    ET.SubElement(cam, 'parent', link='panda_hand')
    ET.SubElement(cam, 'child', link='camera_optical_frame')
    _set_origin(cam, T_hand_cam)
    root.insert(0, ET.Comment(' Kinematically calibrated by panda_joint_calib. Joint compliance is NOT in this file; '
                              'apply compensate_compliance() to commanded joint angles. '))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='utf-8', xml_declaration=True)


def result_to_dict(result):
    p = result['params']
    return dict(
        mode=result['mode'],
        geometry_errors=dict(columns=['tx_m', 'ty_m', 'tz_m', 'rx_rad', 'ry_rad', 'rz_rad'],
                             frames=ARM_JOINTS+['flange'], values=np.asarray(p['geo']).tolist()),
        compliance_rad_per_Nm=dict(zip(ARM_JOINTS, np.asarray(p['compliance']).tolist())),
        T_hand_cam=np.asarray(p['T_hand_cam']).tolist(),
        T_base_board=np.asarray(p['T_base_board']).tolist(),
        intrinsics=dict(zip(INTRINSIC_NAMES, np.asarray(p['intrinsics']).tolist())),
        estimated=result['free'], fixed_redundant=result['fixed_redundant'], std=result['std'],
        rms_px=result['rms_px'], condition=result['condition'], optimizer=result['optimizer'])


def save_result(result, output_dir, urdf=DEFAULT_URDF):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'calibration.json').write_text(json.dumps(result_to_dict(result), indent=2))
    write_urdf(result['params']['geo'], result['params']['T_hand_cam'], output_dir/'panda_calibrated.urdf', urdf)


def load_result(path):
    d = json.loads(Path(path).read_text())
    return dict(geo=np.array(d['geometry_errors']['values']),
                compliance=np.array(list(d['compliance_rad_per_Nm'].values())),
                T_hand_cam=np.array(d['T_hand_cam']), T_base_board=np.array(d['T_base_board']),
                intrinsics=np.array(list(d['intrinsics'].values())))
