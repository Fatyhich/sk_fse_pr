import shutil
import random
import fire
import numpy as np
import cv2

from typing import Tuple, List, Optional, Any
from pathlib import Path
from functools import partial
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from PIL import Image
import os
from src.canguro_processing_tools.utils.parallel_utils import do_parallel
from src.canguro_processing_tools.trajectories.trajectory_2d import Trajectory2D
from src.canguro_processing_tools.utils.camera_utils import project_points, DEFAULT_CAMERA_PARAMS
from src.canguro_processing_tools.extraction.locators import locate_day_dirs_struct
from src.canguro_processing_tools.utils.math_utils import to_relative_frame


_IMAGE_WIDTH = 960
_IMAGE_HEIGHT = 600


def _process_trajectory(traj_path: Path,
                       min_duration: float) -> List[Trajectory2D]:
    full_traj = Trajectory2D.read(traj_path)
    if full_traj.camera_height is None:
        return []
    sub_trajs = full_traj.split_on_completes()
    result = []
    for e in sub_trajs:
        ts = e.poses_timestamps
        duration = (ts[-1] - ts[0]) / 1000.
        if duration >= min_duration:
            result.append(e)
    return result


def _reduce_lists(results: List[List[Trajectory2D]]) -> List[Trajectory2D]:
    reduced = []
    for result in results:
        if len(result) != 0:
            reduced = reduced + result
    return reduced


def _timestamps_to_duration_seconds(start: int, end: int) -> float:
    return (end - start) / 1000.


def _filter_points(points: Optional[np.ndarray]) -> bool:
    # No points fit the projection
    if points is None:
        return False
    
    # Points lie "too high" so it's more likely odometry issue
    if (points[:, 1] < _IMAGE_HEIGHT / 3.).any():
        return False
    
    return True


def _sample_points(trajectory: Trajectory2D, 
                   n_points: int, 
                   points_time_delta: float,
                   output_frames_dir: Path,
                   output_points_dir: Path,
                   use_symlink: bool) -> Tuple[Path, np.ndarray]:
    timestamps = trajectory.frames_timestamps
    time_offset = n_points * points_time_delta
    max_idx = 0
    for i, e in enumerate(timestamps[::-1]):
        if _timestamps_to_duration_seconds(e, timestamps[-1]) > time_offset:
            max_idx = len(timestamps) - i
            break
    
    start_idx = 0

    while start_idx < (max_idx - 1):
        idx = start_idx + 1
        prev_ts = timestamps[start_idx]
        poses = [trajectory.pose_at(prev_ts)]
        frame = trajectory.frame_at(prev_ts)[0]
        frame_ts = prev_ts

        while len(poses) < n_points + 1:
            if _timestamps_to_duration_seconds(prev_ts, timestamps[idx]) >= points_time_delta:
                prev_ts = timestamps[idx]
                poses.append(trajectory.pose_at(prev_ts))
            idx += 1
            if idx >= max_idx:
                break
        
        start_idx = idx

        if len(poses) < 2:
            continue

        poses = np.array(poses)
        poses = to_relative_frame(poses)[1:]
        points = project_points(poses,
                                trajectory.camera_height,
                                DEFAULT_CAMERA_PARAMS,
                                (_IMAGE_WIDTH, _IMAGE_HEIGHT))
        if not _filter_points(points):
            continue

        sample_name = f"{frame.parent.parent.name}__{frame_ts}"
        output_frame = output_frames_dir / f"{sample_name}.{frame.name.split('.')[-1]}"
        output_points = output_points_dir / f"{sample_name}.npy"

        np.save(str(output_points), points)
        if use_symlink:
            output_frame.symlink_to(frame)
        else:
            shutil.copy(str(frame), str(output_frame))


def _sample_frames(input_path: str,
                   output_path: str,
                   overwrite: bool = False,
                   n_workers: int = 0,
                   footsteps_duration: float = 3.,
                   n_footsteps: int = 4,
                   footsteps_time_delta: float = 1.,
                   seed: int = 42,
                   use_symlink: bool = False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.is_dir():
        if not overwrite:
            print(f"{output_path} exists, exiting")
            return
        shutil.rmtree(str(output_path))

    random.seed(seed)
    np.random.seed(seed) 

    # extractions = locate_day_dirs_struct(input_path)
    # trajectories = do_parallel(partial(_process_trajectory, 
    #                                    min_duration = footsteps_duration + 1.),
    #                         extractions,
    #                         n_workers=20,
    #                         use_tqdm=True)
    
    # trajectories = _reduce_lists(trajectories)
    trajectories = _process_trajectory(traj_path=input_path,min_duration=footsteps_duration + 1.)

    output_frames_dir = output_path / "frames"
    output_frames_dir.mkdir(parents=True)
    output_points_dir = output_path / "points"
    output_points_dir.mkdir(parents=True)

    sample_fn = partial(_sample_points,
                        n_points=n_footsteps,
                        points_time_delta=footsteps_time_delta,
                        output_frames_dir=output_frames_dir,
                        output_points_dir=output_points_dir,
                        use_symlink=use_symlink)
    
    do_parallel(sample_fn,
                trajectories,
                n_workers=n_workers,
                use_tqdm=True)


def _split_list(input_list: List[Any], n) -> List[List[Any]]:
    chunk_size, remainder = divmod(len(input_list), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (i < remainder)
        chunks.append(input_list[start:end])
        start = end
    return chunks


def _select_by_score(scores: np.array) -> int:
    return np.argmax(scores) 


def _select_by_area(masks: np.ndarray) -> int:
    return np.argmax([e.sum() for e in masks])


def _run_sam(frame_ids: List[str],
             frames_dir: Path,
             points_dir: Path,
             masks_dir: Path,
             weights_path: Path,
             criterion: str):
    sam = sam_model_registry["vit_h"](checkpoint=weights_path)
    sam = sam.to(device="cuda")
    predictor = SamPredictor(sam)

    for frame_id in tqdm(frame_ids):
        frame = cv2.imread(str(frames_dir / f"{frame_id}.jpg"))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points = np.load(str(points_dir / f"{frame_id}.npy"))
        labels = np.ones((points.shape[0]), dtype=points.dtype)

        predictor.set_image(frame)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        if criterion == "score":
            idx = _select_by_score(scores)
        elif criterion == "area":
            idx = _select_by_area(masks)
        else:
            raise ValueError(f"Unknown criterion {criterion}")
        mask = masks[idx]

        mask = mask.astype(np.uint8)
        Image.fromarray(mask).save(str(masks_dir / f"{frame_id}.png"))


def _generate_masks(input_path: str,
                    criterion: str,
                    weights_path: str,
                    overwrite: bool = False,
                    n_workers: int = 0):
    assert criterion in ("score", "area")
    input_path = Path(input_path)
    output_path = input_path / f"masks_{criterion}"

    if output_path.is_dir():
        if not overwrite:
            print(f"{output_path} exists, exiting")
            return
        shutil.rmtree(str(output_path))
 
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        print(f"{weights_path} does not exits")
        return
    
    frame_ids = [e.stem for e in sorted((input_path / "frames").glob("*.jpg"))]
    if n_workers > 1:
        chunks = _split_list(frame_ids)
    else:
        chunks = [frame_ids]

    output_path.mkdir(parents=True)

    sam_fn = partial(_run_sam,
                     frames_dir=input_path / "frames",
                     points_dir=input_path / "points",
                     masks_dir=output_path,
                     weights_path=weights_path,
                     criterion=criterion)

    do_parallel(sam_fn,
                chunks,
                n_workers=n_workers,
                use_tqdm=False)
    

def _apply_masks(images_path: str, masks_path: str, output_path: str):
    """
    Apply a masks to an images.
    
    Args:
        imagse_path (str): Path to the input images.
        masks_path (str): Path to the masks.
        outputs_path (str): Path to save the masked images.
        
    Returns:
        None
    """
    images_id = os.listdir(images_path)
    masks_id = os.listdir(masks_path)

    # Check if output directory exists, create if not
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_id, mask_id in zip(images_id, masks_id):

        # Make path
        image_path = os.path.join(images_path, image_id)
        mask_path = os.path.join(masks_path, mask_id)
        output_path = os.path.join(output_path, image_id)

        # Load the image and mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure the mask is of the same size as the image
        mask = np.array(mask)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Save the result
        cv2.imwrite(output_path, masked_image)


if __name__ == "__main__":
    fire.Fire({
        "sample_frames": _sample_frames,
        "generate": _generate_masks,
        "apply_masks": _apply_masks,
    })
