import enum
import os
import numpy as np
import cv2
import csv

from typing import Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR, TJPF_BGRA


class AbstractImageIO(ABC):

    @abstractmethod
    def read(self, image_path: Union[str, Path]) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, image: np.ndarray, image_path: Union[str, Path]) -> None:
        pass


class OpenCVImageIO(AbstractImageIO):

    def __init__(self, 
                 read_bgr: bool = True, 
                 write_bgr: bool = True) -> None:
        super(OpenCVImageIO, self).__init__()
        self._read_bgr = read_bgr
        self._write_bgr = write_bgr

    def read(self, image_path: Union[str, Path]) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if not self._read_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def write(self, image: np.ndarray, image_path: Union[str, Path]) -> None:
        if not self._write_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image)


class TurboJPEGImageIO(AbstractImageIO):

    def __init__(self,
                 read_format: str = "bgra", 
                 write_format: str = "bgra") -> None:
        super(TurboJPEGImageIO, self).__init__()
        self._read_format = self._resolve_format(read_format)
        self._write_format = self._resolve_format(write_format)
        self._turbo_jpeg = TurboJPEG()

    def read(self, image_path: Union[str, Path]) -> np.ndarray:
        with open(str(image_path), "rb") as f:
            img = self._turbo_jpeg.decode(f.read(), pixel_format=self._read_format)
        return img

    def write(self, image: np.ndarray, image_path: Union[str, Path]) -> None:
        with open(str(image_path), "wb") as f:
            f.write(self._turbo_jpeg.encode(image, pixel_format=self._write_format))

    def _resolve_format(self, img_format: str):
        if img_format == "bgr":
            return TJPF_BGR
        if img_format == "rgb":
            return TJPF_RGB
        if img_format == "bgra":
            return TJPF_BGRA
        raise ValueError(f"Unknown format {img_format}")


class SequentialCSVWriter:

    def __init__(self, columns: Tuple[str]) -> None:
        self._columns = columns
        self._lines = [self._columns]

    def add_line(self, row: Tuple[str]) -> None:
        self._lines.append(row)

    def dump(self, csv_file: Union[str, Path], clear: bool = True) -> None:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for line in self._lines:
                writer.writerow(line)
        if clear:
            self._lines = [self._columns]


class ImageIO(enum.Enum):
    OPENCV = "opencv"
    TURBO_JPEG = "turbo"


def remove_dir(dir_path: Union[str, Path]):
    dir_path = Path(dir_path)
    if dir_path.is_dir():
        dir_path.rmdir()
