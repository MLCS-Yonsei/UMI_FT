"""
python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

import datetime
import pandas as pd

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        
        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)
        
        # create mapping video if don't exist
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create gripper calibration video if don't exist
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    if mp4_path.name.startswith('map'):
                        continue
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        ###############################################################################################
        # look for video directories in demos
        video_datetime_to_outdir = {}
        video_datetimes = []
        ###############################################################################################

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)


        ###############################################################################################
                # Store the video direction
                video_datetime_to_outdir[start_date] = this_out_dir
                video_datetimes.append(start_date)

        # Sort video datetimes
        video_datetimes.sort()
        ###############################################################################################

        csv_datetimes = []
        ###############################################################################################
        # look for csv file in all subdirectories in input dir
        # input_csv_paths = list(input_dir.glob('**/*.csv')) + list(input_dir.glob('**/*.csv'))
        input_csv_paths = list(input_dir.glob('*.csv'))
        input_csv_paths.sort()
        print(f'Found {len(input_csv_paths)} csv files')
        assert len(input_csv_paths) + 2 == len(input_mp4_paths)
        for i, csv_path in enumerate(input_csv_paths):
            if csv_path.is_symlink():
                print(f"Skipping {csv_path.name}, already moved.")
                continue
            
            # Filter ill rows
            print(f"Filtering {csv_path.name}")
            df = pd.read_csv(csv_path, encoding='latin1')
            numeric_columns = ['timestamp', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'width', 'width_origin', 'minW', 'maxW']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            corrupted_rows = df[df.isnull().any(axis=1)]
            print(f"Found {len(corrupted_rows)} corrupted rows. Dropping them.")
            df_cleaned = df.dropna()
            df_interpolated = df_cleaned.interpolate(method='linear', limit_direction='both', axis=0)
            df_interpolated.to_csv(csv_path, index=False)
            print(f"Filtering succeed")


            # move F/T sensor and gripper width csv file
            # the csv file name are like 'task_YYYYMMDD_HHMMSS.csv'
            csv_filename = csv_path.name
            csv_base_name = csv_filename.split('.')[0]
            name_parts = csv_base_name.split('_')
            if len(name_parts) < 3:
                print(f"csv filename {csv_filename} does not have expected format.")
                continue
            date_part = name_parts[1]
            time_part = name_parts[2]
            csv_datetime_str = date_part + time_part

            try:
                csv_datetime = datetime.datetime.strptime(csv_datetime_str, "%Y%m%d%H%M%S")
                csv_datetimes.append(csv_datetime)
            except ValueError:
                print(f"Could not parse datetime from csv file name {csv_filename}")
                continue
            
            # Find the closest video datetime
            #closest_video_datetime = min(video_datetimes, key=lambda vd: abs(vd - csv_datetime))
            # for vd in video_datetimes
            # vd_diff = abs(vd - csv_datetime)
            # closest_video_datetime = argmin(vd_diff_list) in video_datetimes
            #time_diff = abs(closest_video_datetime - csv_datetime)
            #if time_diff.total_seconds() > 60:
            #    print(f"Warning: csv file {csv_filename} time differs from closest video by {time_diff}")
            #    continue
            
            #this_out_dir = video_datetime_to_outdir[closest_video_datetime]
            this_out_dir = video_datetime_to_outdir[video_datetimes[i + 2]]
            print(f"Pair with {this_out_dir}")
            cfname = 'ft_sensor_gripper_width.csv'
            out_csv_path = this_out_dir.joinpath(cfname)

            shutil.move(csv_path, out_csv_path)

            # Create symlink back from original location
            dots = os.path.join(*['..'] * len(csv_path.parent.relative_to(session).parts))
            rel_path = str(out_csv_path.relative_to(session))
            symlink_path = os.path.join(dots, rel_path)
            csv_path.symlink_to(symlink_path)
        ###############################################################################################

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
