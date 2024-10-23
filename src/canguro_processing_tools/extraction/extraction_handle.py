import json
import pandas as pd

from typing import Union, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Translation:
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class FullPose:
    translation: Translation
    rotation: Quaternion


class ExtractionHandle:

    def __init__(self,
                 extraction_dir: Union[str, Path]) -> None:
        extraction_dir = Path(extraction_dir)

        odometry_df = pd.read_csv(str(extraction_dir / "odometry.csv"))
        # odometry = {
        #     odometry_df["timestamp"][i]: FullPose(
        #         translation=Translation(odometry_df["cart_x"][i],
        #                                 odometry_df["cart_y"][i],
        #                                 odometry_df["cart_z"][i]),
        #         rotation=Quaternion(odometry_df["quat_x"][i],
        #                             odometry_df["quat_y"][i],
        #                             odometry_df["quat_z"][i],
        #                             odometry_df["quat_w"][i])
        #     )
        #     for i in range(len(odometry_df))
        # }
        frames_timestamps = [int(e.stem) for e in sorted((extraction_dir / "frames").glob("*.jpg"))]
        odometry_timestamps = list(odometry_df["timestamp"])

        metadata_file = extraction_dir / "metadata.json"
        camera_height = None
        if metadata_file.is_file():
            with open(str(metadata_file), "r") as f:
                metadata = json.load(f)
            if "height" in metadata:
                camera_height = float(metadata["height"]) / 100.  # Conver cm to m
    
        self._dir_path = extraction_dir
        self._frames_timestamps = frames_timestamps
        self._odometry_timestamps = odometry_timestamps
        self._camera_height = camera_height

    @property
    def dir_path(self) -> Path:
        return self._dir_path

    @property
    def frames_timestamps(self) -> List[int]:
        return self._frames_timestamps
    
    @property
    def odometry_timestamps(self) -> List[int]:
        return self._odometry_timestamps
    
    @property
    def camera_height(self) -> Optional[float]:
        return self._camera_height
