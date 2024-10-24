import time
import threading
import shutil
import fire

from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
from functools import partial
import numpy as np
from src.canguro_processing_tools.extraction.locators import locate_svo_files
from src.canguro_processing_tools.extraction.extractors import (extract_svo_trajectory_safe,
                                                            ImageIO,
                                                            ExtractionOverwritePolicy,
                                                            AbstractExtractionResult)
from src.canguro_processing_tools.utils.parallel_utils import do_parallel
from src.canguro_processing_tools.utils.str_utils import (seconds_to_human_readble_time,
                                                      calculate_zeros_pad,
                                                      zfill_zeros_pad)
from src.canguro_processing_tools.extraction.locators import locate_day_dirs_struct
from src.canguro_processing_tools.extraction.validators import validate_trajectory, TrajectoryStatus
from src.canguro_processing_tools.extraction.extraction_handle import ExtractionHandle
from src.canguro_processing_tools.extraction.statistics import (AbstractExtractionStatistic,
                                                            OdometryTimeStatistic,
                                                            IncompleteCountStatistic,
                                                            NoMetadataStatistic,
                                                            OldNamingStatistic,
                                                            collect_statistics)
from src.canguro_processing_tools.visualization.player_2d_app import Trajectory2dVisualApp
from src.canguro_processing_tools.trajectories.trajectory_2d import Trajectory2D
from src.canguro_processing_tools.utils.sync_utils import FPSBuffer


def _extract_wrap(in_out: Tuple[Path, Path],
                  overwrite: ExtractionOverwritePolicy,
                  image_io: ImageIO,
                  rate: float) -> AbstractExtractionResult:
    return extract_svo_trajectory_safe(in_out[0],
                                       in_out[1],
                                       overwrite,
                                       image_io,
                                       rate)


def _filter_invalid_extraction(extraction_dir: Path) -> Tuple[Path, Optional[ExtractionHandle]]:
    status = validate_trajectory(extraction_dir)
    if status != TrajectoryStatus.INVALID:
        extraction = ExtractionHandle(extraction_dir)
    else:
        extraction = None
    return extraction_dir, extraction


def _section_title(title: str) -> str:
    dashes = "-------------"
    return f"{dashes}{title}{dashes}"


def _print_stat_samples_values(samples_values: List[Tuple[Path, Dict[str, Any]]],
                               statistics: List[AbstractExtractionStatistic],
                               short_names: bool = False):
    for stat in statistics:
        print(_section_title(stat.statistic_name))
        samples = [(e[0], e[1][stat.statistic_id]) for e in samples_values if e[1][stat.statistic_id] is not None]
        if len(samples) == 0:
            print("None")
        else:
            for extraction_dir, stat_value in samples:
                print(stat.build_sample_message(extraction_dir if not short_names else extraction_dir.name,
                                                stat_value))
        print("\n\n\n")


def _print_pooled_values(pooled_values: Dict[str, Any],
                         statistics: List[AbstractExtractionStatistic]):
    print(_section_title("Total stats:"))
    for statistic in statistics:
        print(f"{statistic.statistic_name}: {statistic.build_pooling_message(pooled_values[statistic.statistic_id])}")
    print("\n\n\n")


def _complete_trajectory2d_to_navisets_format(trajectory: Trajectory2D,
                                              output_dir: Path,
                                              rate: float):
    frames_dir = output_dir / "rgb_images"
    frames_dir.mkdir()
    buffer = FPSBuffer(rate)
    frames = []
    poses = []

    for timestamp in trajectory.frames_timestamps:
        src_image = trajectory.frame_at(timestamp)[0]
        pose = trajectory.pose_at(timestamp)
        timestamp, data = buffer.filter(timestamp, (src_image, pose))
        if timestamp is not None:
            frames.append(data[0])
            poses.append(data[1])

    n_zeros = calculate_zeros_pad(len(frames))
    for i, src_image in enumerate(frames):
        target_image = frames_dir / f"{zfill_zeros_pad(i, n_zeros)}.{src_image.name.split('.')[-1]}"
        shutil.copy(str(src_image), str(target_image))
    
    poses = np.array(poses)
    np.save(str(output_dir / "trajectory.npy"), poses)


def _trajectory2d_to_navisets_format(trajectory_path: Path,
                                     output_path: Path,
                                     rate: float):
    trajectory_name = trajectory_path.name
    full_trajectory = Trajectory2D.read(trajectory_path)
    for i, trajectory in enumerate(full_trajectory.split_on_completes()):
        traj_output_dir = output_path / f"{trajectory_name}__{i}"
        traj_output_dir.mkdir(parents=True)
        _complete_trajectory2d_to_navisets_format(trajectory,
                                                  traj_output_dir,
                                                  rate)


def extract_dataset(input_path: str,
                    output_path: str,
                    n_workers: int = 0,
                    overwrite: str = "never",
                    image_io: str = "turbo",
                    fps: float = 15.):
    input_path = Path(input_path)
    output_path = Path(output_path)
    overwrite = ExtractionOverwritePolicy(overwrite)
    image_io = ImageIO(image_io)

    svo_files = locate_svo_files(input_path)
    if len(svo_files) == 0:
        print(f"No .svo or .svo2 files are found with query {input_path}")
        return

    if input_path.is_file():
        # Single SVO file is given as input
        output_path = output_path / svo_files[0].stem
        args = [(svo_files[0], output_path)]
    else:
        # Multiple SVO files are given
        args = [(e, output_path / (e.relative_to(input_path).parent)) for e in svo_files]

    print("Starting extraction...")
    start_time = time.time()
    task_fn = partial(_extract_wrap, 
                      overwrite=overwrite, 
                      image_io=image_io,
                      rate=fps)
    do_parallel(task_fn, args, n_workers=n_workers, use_tqdm=True)
    finish_time = time.time()
    print("Extraction finished")
    print(f"Time elapsed: {seconds_to_human_readble_time(finish_time - start_time)}")


def validate_extraction(input_path: str,
                        raw_reference: str = None,
                        n_workers: int = 0):
    input_path = Path(input_path)
    fps = 30.
    odom_time_stat = OdometryTimeStatistic(fps=fps)
    incomplete_stat = IncompleteCountStatistic(fps=fps)
    no_metadata_stat = NoMetadataStatistic()
    old_naming_stat = OldNamingStatistic()
    all_statistics = [
        odom_time_stat,
        incomplete_stat,
        no_metadata_stat,
        old_naming_stat
    ]
    individual_print_statistics = [
        incomplete_stat,
        no_metadata_stat,
        old_naming_stat
    ]

    extractions = locate_day_dirs_struct(input_path)

    extractions = do_parallel(_filter_invalid_extraction,
                              extractions,
                              n_workers=n_workers,
                              use_tqdm=True)
    valid_extractions = []
    invalid_extractions = []
    for extraction_dir, extraction_handle in extractions:
        if extraction_handle is not None:
            valid_extractions.append(extraction_handle)
        else:
            invalid_extractions.append(extraction_dir)

    samples_values, pooled_values = collect_statistics(valid_extractions,
                                                       all_statistics,
                                                       n_workers=n_workers,
                                                       use_tqdm=False)

    print(_section_title("Invalid extractions:"))
    if len(invalid_extractions) == 0:
        print("No invalid extractions")
    else:
        for e in invalid_extractions:
            print(e)
    print("\n\n\n")

    _print_stat_samples_values(samples_values, individual_print_statistics)
    _print_pooled_values(pooled_values, all_statistics)

    if raw_reference is not None:
        raw_reference = Path(raw_reference)
        reference_dirs = locate_day_dirs_struct(raw_reference)
        missing_dirs = []
        for reference_dir in reference_dirs:
            parts = reference_dir.parts
            extracted_dir = input_path / parts[-2] / parts[-1]
            if not extracted_dir.is_dir():
                missing_dirs.append(reference_dir)
        
        print(_section_title("Missing extractions"))
        if len(missing_dirs) == 0:
            print("No missing extractions")
        else:
            for missing_dir in missing_dirs:
                print(missing_dir)


def extraction_to_navisets_format(input_path: str,
                                  output_path: str,
                                  rate: float = 4.,
                                  n_workers: int = 0):
    extractions = locate_day_dirs_struct(Path(input_path))
    task_fn = partial(_trajectory2d_to_navisets_format,
                      output_path=Path(output_path),
                      rate=rate)
    do_parallel(task_fn,
                extractions,
                n_workers=n_workers,
                use_tqdm=True)


def play_extraction(input_path: str,
                    image_io: str = "turbo"):
    def run_app():
        app = Trajectory2dVisualApp(input_path, ImageIO(image_io))
        app.mainloop()
    
    app_thread = threading.Thread(target=run_app)
    app_thread.start()
    app_thread.join()  # Wait for the thread to finish


if __name__ == "__main__":
    fire.Fire({
        "extract_dataset": extract_dataset,
        "validate_extraction": validate_extraction,
        "play": play_extraction,
        "to_navisets": extraction_to_navisets_format
    })
