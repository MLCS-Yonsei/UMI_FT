"""
python scripts_slam_pipeline/06_generate_dataset_plan_ft_rs_gm.py -i [your_session_directory]
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import pickle
import numpy as np
import json
import math
import collections
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.interpolation_util import get_interp1d, PoseInterpolator

def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = []
    segment_type = []
    for i in range(len(segment_bounds) - 1):
        start, end = segment_bounds[i], segment_bounds[i+1]
        segments.append(slice(start, end))
        segment_type.append(bool_seq[start])
    return segments, np.array(segment_type, dtype=bool)

def pose_interp_from_df(df, start_timestamp=0.0, tx_base_slam=None):
    timestamp_sec = df['timestamp'].to_numpy() + start_timestamp
    cam_pos = df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.eye(4, dtype=np.float32)[None].repeat(len(cam_pos), axis=0)
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()
    if tx_base_slam is not None:
        cam_pose = tx_base_slam @ cam_pose
    return PoseInterpolator(t=timestamp_sec, x=mat_to_pose(cam_pose))

@click.command()
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-o', '--output', default=None)
@click.option('-to', '--tcp_offset', type=float, default=0.14165)
@click.option('-ts', '--tx_slam_tag', default=None)
@click.option('-ml', '--min_episode_length', type=int, default=24)
@click.option('--ignore_cameras', type=str, default=None)
def main(input, output, tcp_offset, tx_slam_tag, min_episode_length, ignore_cameras):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('dataset_plan_rs_gm.pkl')
    else:
        output = pathlib.Path(output)

    tx_cam_tcp = pose_to_mat(np.array([0, 0.086, 0.01465 + tcp_offset, 0, 0, 0]))
    
    if tx_slam_tag is None:
        tx_slam_tag_path = demos_dir.joinpath('mapping', 'tx_slam_tag.json')
    else:
        tx_slam_tag_path = pathlib.Path(tx_slam_tag)
    
    tx_slam_tag_mat = np.array(json.load(open(tx_slam_tag_path))['tx_slam_tag'])
    tx_tag_slam = np.linalg.inv(tx_slam_tag_mat)

    # Stage 1 & 2: Simplified metadata gathering and demo matching
    # In a real implementation, the full logic from the original script should be used here
    # to create video_meta_df and demo_data_list.
    # For this example, we find demos by looking for realsense_data folders.
    demo_dirs = [p.parent for p in demos_dir.glob('**/realsense_data') if p.is_dir()]

    all_plans = []
    for demo_dir in tqdm(demo_dirs, desc="Processing Demos"):
        rs_csv_path = next(demo_dir.joinpath('realsense_data').glob('*.csv'), None)
        if not rs_csv_path:
            continue

        rs_df = pd.read_csv(rs_csv_path)
        demo_timestamps = rs_df['timestamp'].to_numpy()

        # This example assumes one GoPro per demo for simplicity.
        gopro_mp4_path = demo_dir.joinpath('raw_video.mp4')
        if not gopro_mp4_path.is_file():
            continue

        with ExifToolHelper() as et:
            meta = et.get_metadata(str(gopro_mp4_path))[0]
            gopro_start_time = mp4_get_start_datetime(str(gopro_mp4_path)).timestamp()
        
        with av.open(str(gopro_mp4_path), 'r') as container:
            gopro_fps = container.streams.video[0].average_rate

        traj_df = pd.read_csv(demo_dir.joinpath('camera_trajectory.csv'))
        pose_interp = pose_interp_from_df(traj_df[~traj_df['is_lost']], gopro_start_time, tx_tag_slam)

        ft_df = pd.read_csv(demo_dir.joinpath('ft_sensor_gripper_width.csv'))
        ft_time_offset = ft_df['timestamp'].iloc[0] - gopro_start_time
        adj_ft_ts = ft_df['timestamp'].to_numpy() - ft_time_offset

        width_interp = get_interp1d(adj_ft_ts, ft_df['width'].to_numpy())
        force_interp = {axis: get_interp1d(adj_ft_ts, ft_df[axis].to_numpy()) for axis in ['Fx', 'Fy', 'Fz']}
        torque_interp = {axis: get_interp1d(adj_ft_ts, ft_df[axis].to_numpy()) for axis in ['Tx', 'Ty', 'Tz']}

        # Interpolate all data to RealSense timestamps
        valid_mask = (demo_timestamps >= pose_interp.t_min) & (demo_timestamps <= pose_interp.t_max)
        
        # Basic segmentation
        if np.sum(valid_mask) < min_episode_length:
            continue
        
        segments, seg_type = get_bool_segments(valid_mask)
        for s, is_valid in zip(segments, seg_type):
            if not is_valid or (s.stop - s.start) < min_episode_length:
                continue

            start, end = s.start, s.stop
            ts_segment = demo_timestamps[start:end]

            # Create plan for the valid segment
            tcp_pose = pose_interp(ts_segment)
            grippers_plan = [{
                'tcp_pose': tcp_pose,
                'gripper_width': width_interp(ts_segment),
                'Fx': force_interp['Fx'](ts_segment),
                'Fy': force_interp['Fy'](ts_segment),
                'Fz': force_interp['Fz'](ts_segment),
                'Tx': torque_interp['Tx'](ts_segment),
                'Ty': torque_interp['Ty'](ts_segment),
                'Tz': torque_interp['Tz'](ts_segment),
            }]

            gopro_frame_indices = np.round((ts_segment - gopro_start_time) * gopro_fps).astype(int)

            cameras_plan = [
                {
                    'type': 'gopro',
                    'video_path': str(gopro_mp4_path.relative_to(demos_dir.parent)),
                    'frame_indices': gopro_frame_indices.tolist()
                },
                {
                    'type': 'realsense',
                    'path': str(rs_csv_path.parent.relative_to(demos_dir.parent)),
                    'frame_indices': np.arange(start, end).tolist()
                }
            ]

            all_plans.append({
                "episode_timestamps": ts_segment,
                "grippers": grippers_plan,
                "cameras": cameras_plan
            })

    with open(output, 'wb') as f:
        pickle.dump(all_plans, f)
    print(f"Dataset plan saved to {output}")

if __name__ == "__main__":
    main()