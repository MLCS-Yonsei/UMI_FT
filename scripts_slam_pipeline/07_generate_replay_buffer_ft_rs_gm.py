import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

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
    inpaint_tag
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-ms', '--mirror_swap', is_flag=True, default=True)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        try:
            intr_path = next(pathlib.Path(os.path.expanduser(input[0])).absolute().glob('**/gopro_intrinsics_*.json'))
            opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
            fisheye_converter = FisheyeRectConverter(
                **opencv_intr_dict,
                out_size=out_res,
                out_fov=out_fov
            )
        except StopIteration:
            print("Warning: GoPro intrinsics not found. Fisheye conversion will be skipped.")

    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    buffer_start = 0
    gopro_vid_args = defaultdict(list)
    realsense_args = defaultdict(list)

    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan_rs_gm.pkl')
        if not plan_path.is_file(): continue
        
        with open(plan_path, 'rb') as f:
            plan = pickle.load(f)
        
        for episode_idx, plan_episode in enumerate(plan):
            episode_len = len(plan_episode['episode_timestamps'])
            episode_data = {k: v for k, v in plan_episode.items() if k not in ['cameras', 'episode_timestamps']}
            out_replay_buffer.add_episode(data=episode_data, compressors=None)

            for cam_id, camera_plan in enumerate(plan_episode['cameras']):
                if camera_plan['type'] == 'gopro':
                    video_path = str(demos_path.joinpath(camera_plan['video_path']))
                    gopro_vid_args[video_path].append({
                        'camera_idx': cam_id,
                        'frame_indices': camera_plan['frame_indices'],
                        'buffer_start': buffer_start
                    })
                elif camera_plan['type'] == 'realsense':
                    rs_path = str(demos_path.joinpath(camera_plan['path']))
                    realsense_args[rs_path].append({
                        'camera_idx': cam_id,
                        'frame_indices': camera_plan['frame_indices'],
                        'buffer_start': buffer_start
                    })
            buffer_start += episode_len

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    num_cameras = len(plan[0]['cameras']) if plan else 0
    for cam_id in range(num_cameras):
        name = f'camera{cam_id}_rgb'
        out_replay_buffer.data.require_dataset(
            name=name, shape=(buffer_start,) + out_res + (3,),
            chunks=(1,) + out_res + (3,), compressor=img_compressor, dtype=np.uint8
        )

    def process_gopro_frames(replay_buffer, mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            ih, iw = in_stream.height, in_stream.width
            resize_tf = get_image_transform(in_res=(iw, ih), out_res=out_res)
            
            all_needed_frames = set()
            for task in tasks:
                all_needed_frames.update(task['frame_indices'])
            
            decoded_frames = {}
            for frame_idx, frame in enumerate(container.decode(in_stream)):
                if frame_idx in all_needed_frames:
                    decoded_frames[frame_idx] = frame.to_ndarray(format='rgb24')

            for task in tasks:
                img_array = replay_buffer.data[f"camera{task['camera_idx']}_rgb"]
                for i, frame_idx in enumerate(task['frame_indices']):
                    if frame_idx not in decoded_frames: continue
                    img = decoded_frames[frame_idx].copy()
                    # ... (add full inpainting and masking logic here)
                    img = cv2.resize(img, out_res) # Placeholder for transformations
                    img_array[task['buffer_start'] + i] = img

    def process_realsense_frames(replay_buffer, rs_path, tasks):
        for task in tasks:
            cam_name = f"camera{task['camera_idx']}_rgb"
            for i, frame_idx in enumerate(task['frame_indices']):
                img_path = pathlib.Path(rs_path).joinpath(f"{frame_idx}.png")
                if not img_path.exists(): img_path = img_path.with_suffix('.jpg')
                
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, out_res)
                    replay_buffer.data[cam_name][task['buffer_start'] + i] = img

    with tqdm(total=len(gopro_vid_args) + len(realsense_args), desc="Processing Media") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_gopro_frames, out_replay_buffer, path, tasks) for path, tasks in gopro_vid_args.items()}
            futures.update({executor.submit(process_realsense_frames, out_replay_buffer, path, tasks) for path, tasks in realsense_args.items()})

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    print(f"Done! ReplayBuffer saved to {output}")

if __name__ == "__main__":
    main()