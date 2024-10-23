import cv2
import csv
import fire
import time
import logging
import pyzed.sl as sl

from pathlib import Path
from functools import partial
from typing import List
from tqdm.contrib.concurrent import process_map


class Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"
    
NOT_DEFINED = Sentinel('NOT_DEFINED')


def locate_svo_file(input_path: Path) -> List[Path]:
    """
    The distribution of the logical flows of the program.

    If the root folder contains a file, its processing is started.
    If there is a nested directory, a recursive search is started.
    """
    if Path(input_path).is_file() and \
            (input_path.endswith('.svo') or input_path.endswith('.svo2')):
        return [input_path]

    if Path(input_path).is_dir():
        return [path for path in input_path.rglob('*.svo')] \
            + [path for path in input_path.rglob('*.svo2')]
    
    raise ValueError(f"Input path {input_path} is not and .svo/.svo2 file nor directory")


def svo_handler(input_svo_file: Path, output_path_dir: Path) -> str:
    """
    SVO file handler.

    Tool for extracting frames (.jpg) and odom (.csv).


    Parameters:
    input_svo_file (str): Path to the target svo file.
    output_path_dir (str): Path to the directory that will contain the results.
    """
    name = input_svo_file.name
    path = input_svo_file.parts
    input_svo_file = input_svo_file.as_posix()
    output_path = output_path_dir.as_posix()
    path_size = len(path)

    camera_pose = sl.Pose()

    output_dir = f'{output_path}/{path[path_size - 3]}/{path[path_size - 2]}'
    new_dir = Path(output_dir)
    new_dir.mkdir(parents=True)

    columns_names = ['timestamp', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'cart_x', 'cart_y', 'cart_z']
    file_name = f'{output_dir}/odometry.csv'
    lines = [columns_names]

    output_dir_img = Path(output_dir) / "frames"
    output_dir_img.mkdir(parents=True, exist_ok=True)

    init_params = sl.InitParameters(sdk_verbose=0)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.set_from_svo_file(input_svo_file)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logging.error(f'Failed to open Zed object for {input_svo_file}')
        zed.close()
        print('\033[33m [WARNING] An error with sl.Camera occurred, but the program will continue.\033[0m')
        return None

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_2

    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logging.error(f'Failed to enable positional tracking for {input_svo_file}')
        zed.close()
        print('\033[33m [WARNING] An error with Positional Tracking occurred, but the program will continue.\033[0m')
        return None

    left_image = sl.Mat()

    file_counter = 0
    rt_param = sl.RuntimeParameters()
    py_translation = sl.Translation()

    text = None

    print(f"Starting processing {input_svo_file}")

    while True:
        err = zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            filename = output_dir_img / f"{file_counter:08}.jpg"
            cv2.imwrite(str(filename), left_image.get_data())
            tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                rotation = camera_pose.get_orientation().get()
                translation = camera_pose.get_translation(py_translation).get()
                rotation = (rotation[0], rotation[1],
                            rotation[2], rotation[3])
                translation = (translation[0],
                               translation[1],
                               translation[2])
                image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                pose = rotation + translation

                lines.append([image_timestamp] + list(pose))
            else:
                print(f"Wrong position tracking state for {input_svo_file}")
                print(f"State: {tracking_state}, file counter: {file_counter}, timestamp: {zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()}")

            file_counter += 1

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            zed.close()
            text = f'for file {name} created {file_counter} frames'
            break

        else:
            logging.error(f'grab() returned code {err} for {input_svo_file}')
            zed.close()
            print('\033[33m [WARNING] An error with sl.Camera occurred, but the program will continue.\033[0m')
            return None
        
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in lines:
            writer.writerow(line)

    return text


def main(input_path: str = NOT_DEFINED, output_path: str = NOT_DEFINED, n_workers: int = 6):
    """
    Dataset processing tool.

    Util that extract the images and trajectory from svo.

    Parameters:
    source_dir (str): The root directory with sub-folders, containig SVO
    output_path (str): If defined, save results to destination. Else, save localy.
    n_workers (int): You can change the count of Proccesses created
    """
    input_path = Path(input_path)
    print("Creating output folder")

    if output_path == NOT_DEFINED:
        output_path = Path(f'{input_path}_Res')
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    target_files = locate_svo_file(input_path)
    interim_step = partial(svo_handler, output_path_dir=output_path)
    reports = process_map(interim_step, target_files, max_workers=n_workers)

    end_time = time.time()
    print('\n' + '\033[32m'
          + f'Calculating completed and took {end_time - start_time}'
          + '\033[30m' + '\n')

    for report in reports:
        if report:
            print(report)


if __name__ == "__main__":
    fire.Fire(main)
