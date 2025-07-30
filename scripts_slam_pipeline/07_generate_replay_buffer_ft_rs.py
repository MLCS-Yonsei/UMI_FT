# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

from moge.model.v1 import MoGeModel
import torch

import rosbag
from cv_bridge import CvBridge

bridge = CvBridge()


register_codecs()

device = torch.device("cuda")

mogemodel = (MoGeModel.from_pretrained("Ruicheng/moge-vitl").half().to(device).eval())

@torch.no_grad() 
def crop_fisheyeview(img, out_size=(224, 224)):
    h, w = img.shape[:2]
    side = int(min(h, w) / np.sqrt(2))        # largest inscribed square
    y0   = (h - side) // 2                    # top-left corner
    x0   = (w - side) // 2
    masked_side = int(side * 0.9)
    # patch = img[y0:y0 + side, x0:x0 + side]   # crop
    patch = img[y0:y0 + masked_side, x0:x0 + side] 
    if out_size is not None and (side, side) != out_size:
        patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_AREA)
    return patch

@torch.no_grad() 
def get_depth(img, out_size=(224, 224)):
    patch = crop_fisheyeview(img)
    input_tensor = (torch.tensor(patch, dtype=torch.float16)
                      .permute(2, 0, 1)
                      .unsqueeze(0)
                      .to(device)
                      / 255.0)               

    output = mogemodel.infer(input_tensor)

    depth = output["depth"][0].cpu().numpy()
    mask  = output["mask"][0].cpu().numpy()
    del input_tensor, output
    return depth, mask

def depth_to_rgb(depth: np.ndarray,
                 mask:  np.ndarray,
                 cmap:  str = "turbo") -> np.ndarray:

    if mask is not None:
        depth = np.where(mask.astype(bool), depth, np.nan)

    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)
    d_norm = (depth - d_min) / (d_max - d_min + 1e-8)       # 0-1
    d_uint8 = np.nan_to_num(d_norm * 255).astype(np.uint8)   # 0-255

    # OpenCV expects BGR; we'll convert to RGB afterwards.
    cv2_cmap = getattr(cv2, f"COLORMAP_{cmap.upper()}", cv2.COLORMAP_INFERNO)
    bgr = cv2.applyColorMap(d_uint8, cv2_cmap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if mask is not None:
        rgb[~mask.astype(bool)] = 0

    return rgb

# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=True)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    bag_args = list()

    
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl') 
        # dataset_plan = list of dict, each dict represent each video and gripper data
        # {
        # "episode_timestamps": demo_timestamps[start:end] <numpy.arange>
        #
        #  "grippers": dict()
        #               key: tcp_pose, gripper_width, demo_start_pose, demo_end_pose, Fx ~ Tz
        #
        #  "cameras": dict()
        #               key: video_path, video_start_end
        # }
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))

        videos_dict = defaultdict(list)
        bags_dict = defaultdict(list)

        for plan_episode in plan:
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']

                stiffness = gripper['Stiffness']

                Fx = gripper["Fx"]
                Fy = gripper["Fy"]
                Fz = gripper["Fz"]
                force = np.column_stack((Fx, Fy, Fz)).astype(np.float32)

                Tx = gripper["Tx"]
                Ty = gripper["Ty"]
                Tz = gripper["Tz"]
                torque = np.column_stack((Tx, Ty, Tz)).astype(np.float32)


                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
                episode_data[robot_name + '_force'] = force
                episode_data[robot_name + '_torque'] = torque

                episode_data[robot_name + '_stiffness'] = stiffness


            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                bag_path_rel = camera['rosbag_path']
                bag_path = demos_path / bag_path_rel
                assert bag_path.is_file()

                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })

                bags_dict[str(bag_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })

            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        bag_args.extend(bags_dict.items())
        all_videos.update(videos_dict.keys())
    

    print(f"{len(all_videos)} videos used in total!")

    
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

        # dump depth images
        name_depth = f'camera{cam_id}_depth'
        _ = out_replay_buffer.data.require_dataset(
            name=name_depth,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

        name_rs = f'rs_camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name_rs,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def bag_to_zarr(replay_buffer, bag_path, tasks):
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
                start = task["frame_start"]
                end = task["frame_end"]
                bs = task["buffer_start"]
            else:
                assert camera_idx == task['camera_idx']
        name_rs = f'rs_camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name_rs]

        bag = rosbag.Bag(bag_path, "r")
        count = 0
        for topic, msg, _ in bag.read_messages(topics=[f"/camera{camera_idx}/color/image_rect_raw"]):
            if count < start:
                count += 1
                continue
            if count >= end:
                break

            # convert to CV2 BGR then to RGB
            cv_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
            img_array[bs + (count - start)] = rgb
            count += 1

        bag.close()



    def video_to_zarr(replay_buffer, mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]

        name_depth = f'camera{camera_idx}_depth'
        depth_array = replay_buffer.data[name_depth]
        
        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]

                    # get depth image
                    depth, mask = get_depth(img)
                    depth_rgb_img = depth_to_rgb(depth, mask)


                    # compress image
                    img_array[buffer_idx] = img
                    depth_array[buffer_idx] = depth_rgb_img
                    buffer_idx += 1

                    if buffer_idx % 200 == 0:
                        torch.cuda.empty_cache()
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))
            
            for bag_path, tasks in bag_args:
                futures.add(executor.submit(bag_to_zarr,
                    out_replay_buffer, bag_path,  tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")



# %%
if __name__ == "__main__":
    main()
