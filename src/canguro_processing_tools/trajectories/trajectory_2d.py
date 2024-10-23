from __future__ import annotations
import json
import pandas as pd
import numpy as np

from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from canguro_processing_tools.utils.math_utils import quaternion_to_yaw, align_timestamps



class Trajectory2D:

    FORMAT_V1 = 1
    FORMAT_V2 = 2

    def __init__(self,
                 frames: Dict[int, Tuple[Path, Path]],
                 poses: Dict[int, Tuple[float, float, float]],
                 format_version: int,
                 camera_height: Optional[float] = None) -> None:
        if format_version == 2:
            for e in frames.values():
                assert e[0] is not None and e[1] is not None, f"For v2 format, both left and right frames must be present"
        if camera_height is not None:
            camera_height = float(camera_height)
            assert camera_height > 0., f"camera_height must be > 0, got {camera_height}"

        if len(frames) != len(poses):
            if len(frames) < len(poses):
                raise ValueError("Case when len(frames) < len(poses) is not supported")
            is_complete = False
        else:
            frames_ts = sorted(frames.keys())
            poses_ts = sorted(poses.keys())
            if len(frames_ts) != len(poses_ts):
                is_complete = False
            else:
                is_complete = True
                for frame_ts, pose_ts in zip(frames_ts, poses_ts):
                    if frame_ts != pose_ts:
                        is_complete = False
                        break

        self._frames = frames
        self._poses = poses
        self._is_complete = is_complete
        self._format_version = format_version
        self._camera_height = camera_height

    @staticmethod
    def read(trajectory_dir: Union[str, Path]) -> Trajectory2D:
        trajectory_dir = Path(trajectory_dir)
        frames_dir = trajectory_dir / "frames"
        odometry_file = trajectory_dir / "odometry.csv"
        metadata_file = trajectory_dir / "metadata.json"
        if not (frames_dir.is_dir() and odometry_file.is_file()):
            raise ValueError(f"Missing frames dir and/or odometry file in {trajectory_dir}")
        
        odometry_df = pd.read_csv(str(odometry_file))

        poses = {
            odometry_df["timestamp"][i]: (odometry_df["cart_x"][i],
                                          odometry_df["cart_y"][i],
                                          quaternion_to_yaw(odometry_df["quat_x"][i],
                                                odometry_df["quat_y"][i],
                                                odometry_df["quat_z"][i],
                                                odometry_df["quat_w"][i]))
            for i in range(len(odometry_df))
        }

        camera_height = None
        if metadata_file.is_file():
            with open(str(metadata_file), "r") as f:
                metadata = json.load(f)
            if "height" in metadata:
                camera_height = float(metadata["height"]) / 100. # cm to m

        if not (frames_dir / "left").is_dir():
            # V1 format
            format_version = Trajectory2D.FORMAT_V1
            frames = {int(e.stem): (e, None) for e in frames_dir.glob("*.jpg")}
        else:
            format_version = Trajectory2D.FORMAT_V2
            frames = {}
            left_dir = frames_dir / "left"
            right_dir = frames_dir / "right"
            assert right_dir.is_dir(), f"For V2 format, both left and right dirs must be present for {trajectory_dir}"
            left_frames = sorted(left_dir.glob("*.jpg"))
            right_frames = sorted(right_dir.glob("*.jpg"))
            assert len(left_frames) == len(right_frames), f"Left and right frames numbers mismatch {trajectory_dir}"
            for left_frame, right_frame in zip(left_frames, right_frames):
                left_timestamp = left_frame.stem
                right_timestamp = right_frame.stem
                assert left_timestamp == right_timestamp, f"Frame timestamps mismatch for {trajectory_dir}"
                frames[int(left_timestamp)] = (left_frame, right_frame)

        return Trajectory2D(frames, poses, format_version, camera_height)
 
    @property
    def frames_left(self) -> List[Path]:
        return [e[0] for e in sorted(self._frames.values())]
    
    @property
    def frames_right(self) -> Optional[List[Path]]:
        if self._format_version == Trajectory2D.FORMAT_V1:
            return None
        return [e[1] for e in sorted(self._frames.values())]

    @property
    def poses(self) -> np.ndarray:
        return np.array([self._poses[e] for e in sorted(self._poses.keys())])

    @property
    def camera_height(self) -> Optional[float]:
        return self._camera_height

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def format_version(self) -> int:
        return self._format_version

    @property
    def frames_timestamps(self) -> List[int]:
        return sorted(self._frames.keys())

    @property
    def poses_timestamps(self) -> List[int]:
        return sorted(self._poses.keys())

    def frame_at(self, timestamp: int) -> Tuple[Path, Path]:
        return self._frames[timestamp]
    
    def pose_at(self, timestamp: int) -> Optional[np.ndarray]:
        if timestamp in self._poses:
            return np.array(self._poses[timestamp])
        return None

    def split_on_completes(self) -> List[Trajectory2D]:
        if self._is_complete:
            return [Trajectory2D(self._frames.copy(),
                                self._poses.copy(),
                                self._format_version,
                                self._camera_height)]
        
        timestamp_chunks = align_timestamps(self.frames_timestamps,
                                           self.poses_timestamps)
        trajectories = [Trajectory2D({e: self._frames[e] for e in chunk},
                                     {e: self._poses[e] for e in chunk},
                                     self._camera_height) for chunk in timestamp_chunks]
        return trajectories
