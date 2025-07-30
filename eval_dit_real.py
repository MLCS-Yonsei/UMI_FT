"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import zarr
import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from act.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait

from umi.real_world.dit_env import DiTEnv

from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_act_obs_dict,
                                                get_real_umi_action)
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.interpolation_util import get_interp1d, PoseInterpolator

import pickle

OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

def get_current_pose(obs, robots_config):
    episode_start_pose = list()
    for robot_id in range(len(robots_config)):
        pose = np.concatenate([
            obs[f'robot{robot_id}_eef_pos'],
            obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1)[-1]
        episode_start_pose.append(pose)
    return episode_start_pose

def post_process(action_normalizer, raw_action):
    return action_normalizer.unnormalize(raw_action)
    

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('-z', '--zarr_path', default=None, help='Path to dataset zarr')
@click.option('-r', '--replay', is_flag=True, default=False, help='Enable replay mode using stored replay buffer.')
@click.option('-n', '--normalizer_path', required=True, type=str, help='Normalizer path for ACT')
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap, zarr_path, replay, normalizer_path):
    
    max_gripper_width = 1 # used for clipping gripper width, in our case it will be max position
    min_gripper_width = 0
    gripper_speed = 0.2 # used for calculating delta_pos
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    sensors_config = robot_config_data['sensors']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    # print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    query_frequency = cfg.chunk_size
    max_ep_length = cfg.max_ep_length

    # Shaered Memeory Manager : inter process communication
    # keystroke counter : capture keyboard events for control
    with SharedMemoryManager() as shm_manager:  
        with KeystrokeCounter() as key_counter, \
            DiTEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                sensors_config=sensors_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,

                # latency
                camera_obs_latency=0.17,

                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                sensor_obs_horizon=cfg.task.shape_meta.obs.robot0_force.horizon,

                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,

                # action
                max_pos_speed=0.25,
                max_rot_speed=0.16,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)


            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
                print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
            
            # Load replay buffer for replay
            if replay:
                if zarr_path is None:
                    print("Please provide a valid zarr_path for replay mode.")
                    return

                zip_store = zarr.ZipStore(zarr_path, mode='a')
                root = zarr.group(zip_store)
                replay_buffer = ReplayBuffer.create_from_group(root)

                num_episodes = len(replay_buffer.episode_ends)
                print(f"Loaded replay buffer with {num_episodes} episodes.")

                episode_idx = np.random.choice(num_episodes)
                print(f"Replaying episode {episode_idx}...")

                pose_data = dict()
                for robot_idx in range(1):
                    pos = ep[f'robot{robot_idx}_eef_pos'][:]
                    rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][:]
                    grip = ep[f'robot{robot_idx}_gripper_width'][:]
                    pose = np.concatenate([pos, rot], axis=-1)
                    tx_tag_tcp = pose_to_mat(pose)
                    tx_tag_robot = tx_left_right
                    tx_robot_tcp = np.linalg.inv(tx_tag_robot) @ tx_tag_tcp
                    tcp_pose = mat_to_pose(tx_robot_tcp)
                    pose_data[f'robot{robot_idx}_tcp_pose'] = tcp_pose


            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # load policy
            policy = workspace.model
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr # relative
            action_pose_repr = cfg.task.pose_repr.action_pose_repr # relative
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            # load normalizer
            # normalizer_path = os.path.join(cfg.output_dir, 'normalizer.pkl')
            normalizer = pickle.load(open(normalizer_path, 'rb'))
            action_normalizer = normalizer['action']

            # device
            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")

            # get initial obs
            obs = env.get_obs()

            # get current pose
            episode_start_pose = get_current_pose(obs = obs, robots_config=robots_config)

            with torch.no_grad():
                # policy.reset() # ACT policy has no reset function

                # process real obs for act
                obs_dict_np = get_real_act_obs_dict(
                    env_obs=obs, 
                    shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                
                
                # set device
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                

                # query policy
                all_actions = policy(obs_dict)

                # Just for warming up (no temporal emsemble)
                t = 0
                raw_action = all_actions[:, t % query_frequency]
                raw_action = raw_action.squeeze(0).detach().to('cpu').numpy()
                action = post_process(action_normalizer, raw_action)

                if isinstance(action, torch.Tensor):
                    print("action is tensor")
                    action = action.detach().cpu().numpy() 

                assert action.shape[-1] == 10 * len(robots_config)

                action = get_real_umi_action(action, obs, action_pose_repr)

                assert action.shape[-1] == 7 * len(robots_config)

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_width'] for gs in gripper_states])
                                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()


                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)
                        
                        elif key_stroke == KeyCode(char='r'):
                            #  I think this is wrong
                            # we have to change raw pose to robotic pose
                            assert replay == True
                            duration = 3.0
                            s = replay_buffer.get_episode_slice(episode_idx)

                            for robot_idx in range(1):
                                pose = pose_data[f'robot{robot_idx}_tcp_pose'][s.start]
                                grip = ep[f'robot{robot_idx}_gripper_width'][s.start]
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            start_t = time.time()
                            episode_data = replay_buffer.get_episode(episode_idx)
                            time.sleep(max(duration - (time.time() - start_t), 0))

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]
                        elif key_stroke == KeyCode(char='x'): # close
                            for robot_idx in control_robot_idx_list:
                                gripper_target_pos[robot_idx] = max_gripper_width
                        elif key_stroke == KeyCode(char='o'): # open
                            for robot_idx in control_robot_idx_list:
                                gripper_target_pos[robot_idx] = min_gripper_width

                    if start_policy:
                        break

                    precise_wait(t_sample)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])
                    
                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command
                    # env.exec_actions(
                    #     actions=[action], 
                    #     timestamps=[t_command_target-time.monotonic()+time.time()],
                    #     compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== replay buffer control loop ==============
                if replay:
                    try:
                        episode_data = replay_buffer.get_episode(episode_idx)
                        s = replay_buffer.get_episode_slice(episode_idx)
                        # pre-compute interpolation
                        data_frequency = 59.94
                        slowdown = 2.0
                        n_data_samples = len(pose_data['robot0_tcp_pose'][s])
                        data_timestamps = np.arange(n_data_samples).astype(np.float32) / data_frequency
                        exec_timestamps = np.arange(int(np.floor(data_timestamps[-1] * frequency * slowdown))) / frequency / slowdown
                        exec_data_idxs = np.round(np.clip(exec_timestamps, 0, data_timestamps[-1]) * data_frequency).astype(np.int32)

                        actions = np.zeros((len(exec_timestamps), 14))
                        for robot_idx in range(2):
                            data_pose = pose_data[f'robot{robot_idx}_tcp_pose'][s]
                            data_pose_interpolator = PoseInterpolator(data_timestamps, data_pose)
                            data_gripper_interpolator = get_interp1d(data_timestamps, episode_data[f'robot{robot_idx}_gripper_width'])
                            exec_pose = data_pose_interpolator(exec_timestamps)
                            exec_grip = data_gripper_interpolator(exec_timestamps)

                            for i in range(len(exec_pose)):
                                solve_table_collision(
                                    ee_pose=exec_pose[i],
                                    gripper_width=exec_grip[i,0],
                                    height_threshold=robots_config[robot_idx]['height_threshold'])

                            actions[:,robot_idx*7:robot_idx*7+6] = exec_pose
                            actions[:,robot_idx*7+6:robot_idx*7+7] = exec_grip

                        # start episode
                        start_delay = 1.0 
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start)
                        # wait for 1/30 sec to get the closest frame actually
                        # reduces overall latency
                        frame_latency = 1/60
                        precise_wait(eval_t_start - frame_latency, time_func=time.time)
                        print("Started!")

                        for iter_idx, _ in enumerate(exec_timestamps):
                            t = iter_idx / frequency
                            t_cycle_start = t_start + t
                            t_cycle_end = t_cycle_start + 1/frequency

                            # pump obs
                            obs = env.get_obs()

                            action = actions[iter_idx]

                            env.exec_actions(
                                actions=[action], 
                                timestamps=[t_cycle_end-time.monotonic()+time.time()])

                            # plot image overlay
                            data_idx = exec_data_idxs[iter_idx]
                            vis_imgs = list()
                            for camera_idx in range(2):
                                img = episode_data[f'camera{camera_idx}_rgb'][data_idx]
                                vis_img = obs[f'camera{camera_idx}_rgb'][-1]
                                match_img = img.astype(np.float32) / 255
                                avg_img = (vis_img + match_img) / 2
                                vis_img = np.concatenate([vis_img, avg_img, match_img], axis=1)
                                vis_imgs.append(vis_img[...,::-1])
                            vis_img = np.concatenate(vis_imgs, axis=0)
                            cv2.imshow('default', vis_img)
                            key_stroke = cv2.pollKey()

                            press_events = key_counter.get_press_events()
                            stop_episode = False
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    # Stop episode
                                    # Hand control back to human
                                    print('Stopped.')
                                    stop_episode = True
                            if stop_episode:
                                env.end_episode()
                                break
                            
                            precise_wait(t_cycle_end)
                        env.end_episode()

                    except KeyboardInterrupt:
                        print("Interrupted!")
                        # stop robot.
                        env.end_episode()

                    print("Stopped.")

                # ========== policy control loop ==============
                try:
                    # start episode
                    # policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current obs
                    obs = env.get_obs()
                    
                    # get current pose
                    episode_start_pose = get_current_pose(obs = obs, robots_config=robots_config)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None


                    # for plotting
                    all_norm_actions = []
                    all_unnorm_actions = []
                    all_real_umi_actions = []
                    all_target_poses = []

                    t = 0
                    # ACT run during max timesteps
                    # while True:
                    for t in range(max_ep_length):
                    # for t in range(100):
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        # t_cycle_end = eval_t_start + (t + 1) * dt
                        
                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()

                            # process real obs for act
                            obs_dict_np = get_real_act_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            
                            # set device
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            # query policy
                            if t % query_frequency == 0:
                                all_actions = policy(obs_dict)
                                # print("all actions: ", all_actions)

                            raw_action = all_actions[:, t % query_frequency]

                            # print("raw action: ", raw_action)

                            raw_action = raw_action.squeeze(0).detach().to('cpu').numpy()
                            all_norm_actions.append(raw_action)

                            raw_action = post_process(action_normalizer, raw_action)
                            # print("unnormalized action: ", action)
                            all_unnorm_actions.append(raw_action)

                            if isinstance(raw_action, torch.Tensor):
                                raw_action = raw_action.detach().cpu().numpy() 

                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            all_real_umi_actions.append(action)
                            # print("real umi action: ", action)
                            print('Inference latency:', time.time() - s)

                            # make one timestep action to (num_robot, action_dim)
                            this_target_poses = action.reshape(1, -1)
                            # print(this_target_poses.shape) # (1, 7)
                            # print(this_target_poses[:, 0:6].shape) # (1, 7)
                            # print(this_target_poses[:, 6].shape) # (1, )
                            # print(this_target_poses.reshape([len(robots_config), -1]).shape) # (1, 7)

                            # assert this_target_poses.shape[1] == len(robots_config) * 7
                            # for robot_idx in range(len(robots_config)):
                            #     solve_table_collision(
                            #         ee_pose=this_target_poses[robot_idx * 7: robot_idx * 7 + 6],
                            #         gripper_width=this_target_poses[robot_idx * 7 + 6],
                            #         height_threshold=robots_config[robot_idx]['height_threshold']
                            #     )
                            # # solve collison between two robots
                            # solve_sphere_collision(
                            #     ee_poses=this_target_poses.reshape([len(robots_config), -1]),
                            #     robots_config=robots_config
                            # )

                            # deal with timing for single action <ACT>
                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            # action_timestamps = np.array([obs_timestamps[-1] + dt])
                            # action_timestamps = np.array([max(time.time() + dt, obs_timestamps[-1] + 2 * dt)])
                            action_timestamps = np.array([time.time() + dt])
                            at = time.time() + dt
                            # print("action timestamp: {:.6f}".format(at))


                            # print(dt)
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            # print("is new: ", is_new)
                            if not np.any(is_new):  # If no valid future actions
                                this_target_poses = this_target_poses[[-1]]  # Use last pose

                                # Schedule action at the next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]
                            all_target_poses.append(this_target_poses)
                            print("execute action: ", this_target_poses)
                            # execute actions
                            # env.exec_actions(

                            env.exec_one_action(
                                action=this_target_poses,
                            )
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

                            # visualize
                            episode_id = env.replay_buffer.n_episodes
                            obs_left_img = obs['camera0_rgb'][-1]
                            obs_right_img = obs['camera0_rgb'][-1]
                            vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('default', vis_img[...,::-1])

                            _ = cv2.pollKey()
                            press_events = key_counter.get_press_events()
                            stop_episode = False
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    # Stop episode
                                    # Hand control back to human
                                    print('Stopped.')
                                    stop_episode = True

                            t_since_start = time.time() - eval_t_start
                            if t_since_start > max_duration:
                                print("Max Duration reached.")
                                stop_episode = True
                            if stop_episode:
                                # env.end_episode()
                                break

                            # wait for execution
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference
                            # if t == max_ep_length:
                            #     break
                            # else:
                            #     t +=1 

                    pickle.dump([all_norm_actions, all_unnorm_actions, all_real_umi_actions, all_target_poses], open('/home/jaco/actions.pkl', 'wb'))

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()