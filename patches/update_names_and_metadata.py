import shutil
import fire
import pandas as pd

from pathlib import Path


def process_record_dir(record_dir: Path, root_dir: Path, raw_dir: Path):
    src_metadata = raw_dir / record_dir.relative_to(root_dir) / "metadata.json"
    target_metadata = record_dir / "metadata.json"
    if src_metadata.is_file():
        if target_metadata.is_file():
            target_metadata.unlink()
        shutil.copy(str(src_metadata), str(target_metadata))

    trajectory = pd.read_csv(str(record_dir / "odometry.csv"))
    frames = sorted((record_dir / "frames").glob("*.jpg"))
    if len(frames) != len(trajectory):
        print(f"Frames-odometry mismatch for {record_dir}")
        return
    
    for i in range(len(frames)):
        timestamp_name = f"000{trajectory['timestamp'][i]}.jpg"
        frame = frames[i]
        new_frame = frame.parent / timestamp_name
        frame.rename(new_frame)


def process_day_dir(day_dir: Path, root_dir: Path, raw_dir: Path):
    date_name = day_dir.name
    if len(date_name.split("_")) != 3:
        print(f"{day_dir} does not follow format, skipping")

    record_dirs = sorted(day_dir.glob("*/"))

    for record_dir in record_dirs:
        process_record_dir(record_dir, root_dir, raw_dir)


def main(input_dir: str = "/mnt/vol0/datasets/egowalk/extracted",
         raw_dir: str = "/home/timur_akht/egowalk_raw_mnt/"):
    input_dir = Path(input_dir)
    raw_dir = Path(raw_dir)
    day_dirs = sorted(input_dir.glob("*/"))
    for day_dir in day_dirs:
        process_day_dir(day_dir, input_dir, raw_dir)


if __name__ == "__main__":
    fire.Fire(main)
