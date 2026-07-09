import numpy as np
from lietorch import SE3
from pathlib import Path
from pose_gaussian.dataset.dataset_readers import readBlenderInfo
from scipy.spatial.transform import Rotation

num_views = 75
path = Path("/workspace/data/noisy_dataset-rot0_03-trans1")
dirnames = [f.name for f in path.iterdir() if f.is_dir()]
last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

f = open('pose_error.csv', 'w')
f.write('scene, position, orientation\n')
total_position_error = 0
total_angle_error = 0
cnt = 0
for dirname in sorted(dirnames):
    if 'ours' in dirname or 'r2gs' in dirname:
        continue
    path_q_gt = path / dirname / "vol_gt_cone/quaternions.npy"
    path_t_gt = path / dirname / "vol_gt_cone/translations.npy"
    path_ours = path / "ours_results" / dirname / "eval/iter_030000"

    dq_gt = np.load(str(path_q_gt), allow_pickle=True).item()
    dt_gt = np.load(str(path_t_gt), allow_pickle=True).item()
    source_path = f'/workspace/data/noisy_dataset-rot0_03-trans1/{dirname}/vol_gt_cone'
    scene_info = readBlenderInfo(source_path, False)
    camera_info = scene_info.train_cameras

    angle_error = 0
    position_error = 0
    for i in range(num_views):
        # if i < 30:
        #     continue
        # calc init camera pose
        camera = camera_info[i]
        R_init = camera.R.T
        t_init = camera.T
        T_init = np.hstack((R_init, t_init[:, None]))
        T_init = np.vstack((T_init, last_row))
        
        # calc gt noise
        dq_gt_i = dq_gt[f'proj_train_{i:04d}']
        dR_gt_i = Rotation.from_quat(dq_gt_i).as_matrix()
        dt_gt_i = dt_gt[f'proj_train_{i:04d}'] / 128
        
        dT_gt_i = np.hstack((dR_gt_i, dt_gt_i[:, None]))
        dT_gt_i = np.vstack((dT_gt_i, last_row))

        # dT_gt_i = dT_prefix @ dT_gt_i
        # dT_gt_i = dT_gt_i @ dT_postfix

        #calc estimated pose
        fname = f"cam_calib_result_proj_train_{i:04d}.npz"
        pose_estimate = np.load(str(path_ours / fname), allow_pickle=True)

        w, x, y, z = pose_estimate['quaternion']
        q = np.array([x, y, z, w])
        R_est = Rotation.from_quat(q).as_matrix()
        t_est = pose_estimate['translation']
        T_est = np.hstack((R_est, t_est[:, None]))
        T_est = np.vstack((T_est, last_row))
        
        error_T = np.linalg.inv(T_init @ np.linalg.inv(dT_gt_i)) @ T_est
        error_t = error_T[:3, 3]
        error_R = error_T[:3, :3]

        angle_error += (np.arccos((np.trace(error_R) - 1) / 2) * 180 / np.pi)**2
        position_error += np.sum(error_t**2)
    angle_error = np.sqrt(angle_error / num_views)
    position_error = np.sqrt(position_error / num_views)
    print(dirname, angle_error, position_error)
    total_position_error += position_error
    total_angle_error += angle_error
    cnt += 1
    f.write(f'{dirname}, {position_error:e}, {angle_error:e}\n')
total_angle_error /= cnt
total_position_error /= cnt

f.write(f'mean, {total_position_error:e}, {total_angle_error:e}')
f.close()