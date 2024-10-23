import csv
import fire
import time
import logging
import datetime
import pyzed.sl as sl

from pathlib import Path
from typing import Union, List
from tqdm.contrib.concurrent import process_map

class Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"
    
NOT_DEFINED = Sentinel('NOT_DEFINED')


def locate_svo_file(input_path: Union[str, Path]) -> List[str]:
    """
    The distribution of the logical flows of the program.

    If the root folder contains a file, its processing is started.
    If there is a nested directory, a recursive search is started.
    """
    input_path = Path(input_path)

    if Path(input_path).is_file() and \
            (input_path.endswith('.svo') or input_path.endswith('.svo2')):
        return [input_path]

    if Path(input_path).is_dir():
        return [path for path in input_path.rglob('*.svo')] \
            + [path for path in input_path.rglob('*.svo2')]
    
    raise ValueError(f"Input path {input_path} is not and .svo/.svo2 file nor directory")


def seconds_to_human_readble_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = round(seconds % 60)

    return f"{hours:02}h {minutes:02}m {remaining_seconds:02}s"


def calc_duration(file_path: Union[str, Path]):
    """Little util to count duration of recorded video."""
    # Configure logging
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Specify SVO path parameter
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                    coordinate_units=sl.UNIT.METER,
                                    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
                                    sdk_verbose=0)
    init_params.set_from_svo_file(str(file_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    name = file_path.stem
    # Create ZED objects
    zed = sl.Camera()

    try:
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print('\033[31m [ERROR] ' + str(err) + '\033[0m' + f' -> {file_path}')
            zed.close()
            return None
    except Exception as e:
        # Log the exception
        logging.error(f'An exception occurred: {str(e)}')
        zed.close()
        # Continue with the program
        print('\033[33m [WARNING] An error occurred, but the program will continue.\033[0m')
        return None

    nb_frames = zed.get_svo_number_of_frames()
    camera_fps = zed.get_camera_information().camera_configuration.fps

    # Convert to datetime and format
    first_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()
    readable_time = datetime.datetime.fromtimestamp(first_image_timestamp).strftime('%B_%Y_%d_%H:%M:%S')

    res = round(nb_frames / camera_fps, 2)
    zed.close()

    text = (name, first_image_timestamp, readable_time, res)
    return text


def main(input_path: str = "/mnt/vol0/datasets/egowalk/raw", output_file: str = NOT_DEFINED, n_workers: int = 6):
    """
    Util to count the duration of svo file.

    Parameters:
    input_path (str): The root directory with sub-folders, containig SVO
    output_file (bool): If specified, then output wiil be duplicated to file
    n_workers (int): You can change the count of Proccesses created
    """
    assert input_path, "[ERROR] input directory must be defined"

    target_files = locate_svo_file(input_path)

    start_time = time.time()
    print('\033[32m' + f'Calculating with {n_workers} workers, please wait for result' + '\033[0m\n')

    # procces_map is wrapper around ProccesPoolExecutor map method
    durations = process_map(calc_duration, target_files, max_workers=n_workers)

    end_time = time.time()
    print('\n' + '\033[32m'
          + f'Calculating completed and took {round(end_time - start_time, 2)}'
          + '\033[0m' + '\n')

    total_duration = 0
    for duration in durations:
        if duration:
            print(f'{duration[0]}   ' + seconds_to_human_readble_time(duration[3]))
            total_duration += duration[3]
        else:
            print('Skipped broken file')
            continue

    print("----------------------------------")
    print('Total duration:       ' + seconds_to_human_readble_time(total_duration))

    if output_file != NOT_DEFINED:
        columns_names = ['filename', 'timestamp', 'human_date_time', 'duration_seconds']
        file_name = f'{Path(output_file).stem}.csv'
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns_names)
            for duration in durations:
                if duration:
                    writer.writerow(duration)
                else:
                    continue


if __name__ == '__main__':
    fire.Fire(main)
