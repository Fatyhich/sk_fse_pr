import enum
import pandas as pd

from typing import Union
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from .canguro_processing_tools.utils.math_utils import align_timestamps


class TrajectoryStatus(enum.Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


def validate_trajectory(trajectory_path: Union[str, Path]) -> TrajectoryStatus:
    trajectory_path = Path(trajectory_path)

    frames_dir = trajectory_path / "frames"
    if not frames_dir.is_dir():
        return TrajectoryStatus.INVALID
    
    odometry_file = trajectory_path / "odometry.csv"
    if not odometry_file.is_file():
        return TrajectoryStatus.INVALID
    
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) == 0:
        return TrajectoryStatus.INVALID
    
    odometry = pd.read_csv(str(odometry_file))
    if len(odometry) != len(frames):
        return TrajectoryStatus.INCOMPLETE
    
    return TrajectoryStatus.COMPLETE
