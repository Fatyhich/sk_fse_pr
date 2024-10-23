import shutil
import fire
import pandas as pd

from pathlib import Path


def validate_recording(record_dir: Path) -> bool:
    odometry_file = record_dir / "odometry.csv"
    if not odometry_file.is_file():
        print(f"Odometry file is missing for {record_dir}")
        return False
    frames_dir = record_dir / "frames"
    if not frames_dir.is_dir():
        print(f"Frames dir is missing for {record_dir}")
        return False
    trajectory = pd.read_csv(str(odometry_file))
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(trajectory) != len(frames):
        print(f"Number of frames ({len(frames)}) does not match length of the odometry ({len(trajectory)}) for {record_dir}")
        return False
    return True
    

def process_record_dir(record_dir: Path):
    datetime_name = record_dir.name
    if "__" in datetime_name:
        print(f"Dir {record_dir} seems already to be reformatted, skipping")
        return
    
    if not validate_recording(record_dir):
        shutil.rmtree(str(record_dir))
        return
    
    day, month, year, hours, minutes, seconds = datetime_name.split("_")
    new_datetime_name = f"{year}_{month}_{day}__{hours}_{minutes}_{seconds}"
    new_dir = record_dir.parent / new_datetime_name
    record_dir.rename(new_dir)


def process_day_dir(day_dir: Path):
    date_name = day_dir.name
    if len(date_name.split("_")) != 3:
        print(f"{day_dir} does not follow format, skipping")

    record_dirs = sorted(day_dir.glob("*/"))
    if len(record_dirs) == 0:
        shutil.rmtree(day_dir)
        return
    for record_dir in record_dirs:
        process_record_dir(record_dir)
    
    day, month, year = date_name.split("_")
    if day == "2024":
        return
    
    new_date_name = f"{year}_{month}_{day}"
    new_day_dir = day_dir.parent / new_date_name

    day_dir.rename(new_day_dir)


def main(input_dir: str = "/mnt/vol0/datasets/egowalk/extracted"):
    input_dir = Path(input_dir)
    day_dirs = sorted(input_dir.glob("*/"))
    for day_dir in day_dirs:
        process_day_dir(day_dir)


if __name__ == "__main__":
    fire.Fire(main)
