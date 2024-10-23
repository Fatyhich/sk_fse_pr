import time
import enum
import shutil
import traceback
import cv2

try:
    import pyzed.sl as sl
except:
    print("Warning: ZED SDK (pyzed) is not installed, some functions are unavailable")

from typing import Union, List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from canguro_processing_tools.utils.io_utils import (AbstractImageIO, 
                                               OpenCVImageIO, 
                                               TurboJPEGImageIO, 
                                               SequentialCSVWriter,
                                               ImageIO,
                                               remove_dir)
from canguro_processing_tools.extraction.validators import (validate_trajectory, 
                                                 TrajectoryStatus)
from canguro_processing_tools.utils.str_utils import seconds_to_human_readble_time
from canguro_processing_tools.utils.sync_utils import FPSBuffer


class AbstractExtractionResult:
    
    def __init__(self, input_file: Union[str, Path]) -> None:
        self._input_file = Path(input_file)

    @property
    def input_file(self) -> Path:
        return self._input_file
    
    @abstractmethod
    def __str__(self) -> str:
        pass


class SuccessfulExtractionResult(AbstractExtractionResult):

    def __init__(self,
                 input_file: Union[str, Path],
                 output_dir: Union[str, Path],
                 messages: List[str] = None) -> None:
        super(SuccessfulExtractionResult, self).__init__(input_file)
        self._output_dir = Path(output_dir)
        if messages is None:
            self._messages = []
        else:
            self._messages = [e for e in messages]

    @property
    def output_dir(self) -> Path:
        return self._output_dir
    
    @property
    def messages(self) -> List[str]:
        return self._messages
    
    def __str__(self) -> str:
        info_string = f"Succesfully extracted {self._input_file} to {self._output_dir}"
        if len(self._messages) != 0:
            info_string = info_string + "\nMessages:"
            for message in self._messages:
                info_string = info_string + "\n" + message
        return info_string


class FailedExtractionResult(AbstractExtractionResult):

    def __init__(self, 
                 input_file: str | Path,
                 message: str,
                 stack_trace: str | None = None) -> None:
        super(FailedExtractionResult, self).__init__(input_file)
        self._message = message
        self._stack_trace = stack_trace

    @property
    def message(self) -> str:
        return self._message
    
    @property
    def stack_trace(self) -> str | None:
        return self._stack_trace
    
    def __str__(self) -> str:
        info_string = f"Exctraction failed for {self._input_file}\nError message: {self._message}"
        if self._stack_trace is not None:
            info_string = info_string + "\n" + self._stack_trace
        return info_string


class SkippedExtractionResult(AbstractExtractionResult):

    def __init__(self, input_file: str | Path) -> None:
        super(SkippedExtractionResult, self).__init__(input_file)

    def __str__(self) -> str:
        return f"Extraction skipped for {self._input_file}"


class ExtractionOverwritePolicy(enum.Enum):
    ALL = "all"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    NEVER = "never"


def resolve_overwrite(policy: ExtractionOverwritePolicy,
                      validation_result: TrajectoryStatus) -> bool:
    if policy == ExtractionOverwritePolicy.ALL:
        return True
    if policy == ExtractionOverwritePolicy.INVALID:
        return validation_result == TrajectoryStatus.INVALID
    if policy == ExtractionOverwritePolicy.INCOMPLETE:
        return validation_result == TrajectoryStatus.INCOMPLETE or validation_result == TrajectoryStatus.INVALID
    if policy == ExtractionOverwritePolicy.NEVER:
        return False
    raise ValueError(f"Unknown policy {policy}")


def extract_svo_trajectory(input_svo_file: Union[str, Path], 
                           output_dir: Union[str, Path],
                           overwrite_policy: ExtractionOverwritePolicy,
                           image_io: ImageIO,
                           rate: Optional[float]) -> AbstractExtractionResult:
    start_time = time.time()
    input_svo_file = Path(input_svo_file)
    output_dir = Path(output_dir)
    fps_buffer = FPSBuffer(rate)

    if image_io == ImageIO.OPENCV:
        image_writer = OpenCVImageIO()
    elif image_io == ImageIO.TURBO_JPEG:
        image_writer = TurboJPEGImageIO(read_format="bgra",
                                        write_format="bgra")
    else:
        raise ValueError(f"Unknown image IO method {image_io}")
    csv_writer = SequentialCSVWriter(columns=('timestamp', 
                                              'quat_x', 'quat_y', 'quat_z', 'quat_w', 
                                              'cart_x', 'cart_y', 'cart_z'))
    
    messages = []

    if output_dir.is_dir():
        validation = validate_trajectory(output_dir)
        if resolve_overwrite(overwrite_policy, validation):
            msg = f"Removing {validation} trajectory {output_dir} accoring to the '{overwrite_policy}' policy"
            print(msg)
            messages.append(msg)
            shutil.rmtree(str(output_dir))
        else:
            msg = f"Skipping {validation} trajectory {output_dir} accoring to the '{overwrite_policy}' policy"
            print(msg)
            return SkippedExtractionResult(input_svo_file)

    init_params = sl.InitParameters(sdk_verbose=0)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.set_from_svo_file(str(input_svo_file))
    init_params.svo_real_time_mode = False
    camera_pose = sl.Pose()
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        msg = f"Failed to open Zed object for {input_svo_file}, error code: {err}"
        print(msg)
        zed.close()
        return FailedExtractionResult(input_svo_file, msg)

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_2
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        msg = f"Failed to enable positional tracking for {input_svo_file}, error code: {err}"
        print(msg)
        zed.close()
        return FailedExtractionResult(input_svo_file, msg)

    output_dir.mkdir(parents=True, exist_ok=False)
    frames_output_dir = output_dir / "frames"
    frames_output_dir.mkdir(parents=False, exist_ok=False)

    frames_output_dir_left = frames_output_dir / "left"
    frames_output_dir_left.mkdir(parents=False, exist_ok=False)
    frames_output_dir_right = frames_output_dir / "right"
    frames_output_dir_right.mkdir(parents=False, exist_ok=False)

    left_image = sl.Mat()
    right_image = sl.Mat()
    file_counter = 0
    rt_param = sl.RuntimeParameters()
    py_translation = sl.Translation()
    tracker_warnings = set()

    print(f"Starting processing {input_svo_file}")

    while True:
        err = zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:

            image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            
            tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                rotation = camera_pose.get_orientation().get()
                translation = camera_pose.get_translation(py_translation).get()
                rotation = (rotation[0], rotation[1],
                            rotation[2], rotation[3])
                translation = (translation[0],
                               translation[1],
                               translation[2])
                pose = rotation + translation
            else:
                if tracking_state not in tracker_warnings:
                    msg = f"Positional tracking error detected for {input_svo_file}, error code: {tracking_state} (warned once)"
                    print(msg)
                    messages.append(msg)
                    tracker_warnings.add(tracking_state)
                pose = None

            timestamp_to_write, data = fps_buffer.filter(image_timestamp, (left_image.get_data(),
                                                                           right_image.get_data(),
                                                                           pose))
            if data is not None:
                left_img_to_write = data[0]
                right_img_to_write = data[1]
                pose_to_write = data[2]

                image_writer.write(left_img_to_write, 
                                   frames_output_dir_left / f"000{timestamp_to_write}.jpg")
                image_writer.write(right_img_to_write,
                                   frames_output_dir_right / f"000{timestamp_to_write}.jpg")
                
                if pose_to_write is not None:
                    csv_writer.add_line((timestamp_to_write,) + pose_to_write)

                file_counter += 1

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            zed.close()
            break

        else:
            msg = f"grab() returned code {err} for {input_svo_file}"
            print(msg)
            zed.close()
            return FailedExtractionResult(input_svo_file, msg)
        
    csv_writer.dump(output_dir / "odometry.csv")

    src_metadata = input_svo_file.parent / "metadata.json"
    if src_metadata.is_file():
        target_metadata = output_dir / src_metadata.name
        shutil.copy(str(src_metadata), str(target_metadata))
    else:
        msg = f"Metdata not found for {input_svo_file}"
        print(msg)
        messages.append(msg)

    finish_time = time.time()
    time_elapsed = seconds_to_human_readble_time(finish_time - start_time)
    video_time = seconds_to_human_readble_time(int(round(file_counter / (rate if rate is not None else 30))))
    msg = f"Finished {input_svo_file} in {time_elapsed}, estimated video time is {video_time}"
    messages.append(msg)
    print(msg)

    return SuccessfulExtractionResult(input_svo_file,
                                      output_dir,
                                      messages)


def extract_svo_trajectory_safe(input_svo_file: Union[str, Path], 
                                output_dir: Union[str, Path],
                                overwrite_policy: ExtractionOverwritePolicy,
                                image_io: ImageIO,
                                rate: Optional[float]) -> AbstractExtractionResult:
    try:
        return extract_svo_trajectory(input_svo_file,
                                      output_dir,
                                      overwrite_policy,
                                      image_io,
                                      rate)
    except Exception as e:
        print(e)
        return FailedExtractionResult(input_svo_file,
                                      str(e),
                                      traceback.format_exc())
