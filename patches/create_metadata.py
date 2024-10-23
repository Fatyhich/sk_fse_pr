import fire
import json
import numpy as np
import pandas as pd

from pathlib import Path


def check_missing_metadata(dirs: list[Path]) -> bool:
    has_missing = False
    for directory in dirs:
        if len(sorted(directory.glob("*.csv"))) == 4 and len(sorted(directory.glob("*.svo2"))) == 1:
            if not (directory / "metadata.json").is_file():
                has_missing = True
    return has_missing


def main(input_dir: str = "/home/timur_akht/egowalk_raw_mnt", all: bool = False):
    input_dir = Path(input_dir)
    day_dirs = sorted(input_dir.glob("*/"))

    for day_dir in day_dirs:
        if day_dir.name.startswith("2024"):
            year, month, day = day_dir.name.split("_")
        else:
            year, month, day = day_dir.name.split("_")[::-1]

        date_time_dirs = sorted(day_dir.glob("*/"))
        if check_missing_metadata(date_time_dirs) or all:
            print(f"Height for the day {day}.{month}.{year}")
            height = input()
            if height == "":
                continue
            height = float(height)
            print(f"Location for the day {day}.{month}.{year}")
            location = str(input())

            for date_time_dir in day_dir.glob("*/"):
                if len(sorted(date_time_dir.glob("*.csv"))) == 4 and len(sorted(date_time_dir.glob("*.svo2"))) == 1:
                    with open(str(date_time_dir / "metadata.json"), "w") as f:
                        json.dump({"height": height, "location": location}, f, indent=2)



if __name__ == "__main__":
    fire.Fire(main)
