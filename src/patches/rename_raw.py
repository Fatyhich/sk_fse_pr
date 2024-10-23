import fire

from pathlib import Path


TARGET_FILES = [
    ("-acceleration-covariance", "csv"),
    ("-angular-speed-covariance", "csv"),
    ("-imu-data", "csv"),
    ("-pose-covariance", "csv"),
    ("", "svo2")
]


def rename_file(root_dir: Path, 
                datetime_name: str, 
                new_datetime_name: str, 
                extension: str, 
                postfix: str = "") -> tuple[Path | None, Path | None]:
    old_file_name = f"{datetime_name}{postfix}.{extension}"
    old_file = root_dir / old_file_name
    if not old_file.is_file():
        print(f"Warning: file {old_file} does not exist")
        return None, None
    new_file_name = f"{new_datetime_name}{postfix}.{extension}"
    new_file = root_dir / new_file_name
    if new_file.is_file():
        print(f"Warning: file {new_file} already exists")
        return None, None
    return old_file, new_file


def process_record_dir(record_dir: Path):
    datetime_name = record_dir.name
    if "__" in datetime_name:
        print(f"Dir {record_dir} seems already to be reformatted, skipping")
        return
    
    day, month, year, hours, minutes, seconds = datetime_name.split("_")
    new_datetime_name = f"{year}_{month}_{day}__{hours}_{minutes}_{seconds}"
    
    to_rename = []

    for target_file in TARGET_FILES:
        old_file, new_file = rename_file(root_dir=record_dir,
                                         datetime_name=datetime_name,
                                         new_datetime_name=new_datetime_name,
                                         extension=target_file[1],
                                         postfix=target_file[0])
        if old_file is None or new_file is None:
            print(f"Skipping {record_dir}")
            return
        else:
            to_rename.append((old_file, new_file))

    for old_file, new_file in to_rename:
        # print(f"Renaming {old_file} to {new_file}")
        old_file.rename(new_file)

    new_dir = record_dir.parent / new_datetime_name
    # print(f"Renaming dir {record_dir} to {new_dir}")
    try:
        record_dir.rename(new_dir)
    except Exception as e:
        print(f"Failed to rename {record_dir} to {new_dir}")
        print(e)


def process_day_dir(day_dir: Path):
    date_name = day_dir.name
    if len(date_name.split("_")) != 3:
        print(f"{day_dir} does not follow format, skipping")

    record_dirs = sorted(day_dir.glob("*/"))
    for record_dir in record_dirs:
        process_record_dir(record_dir)
    
    day, month, year = date_name.split("_")
    if day == "2024":
        return
    
    new_date_name = f"{year}_{month}_{day}"
    new_day_dir = day_dir.parent / new_date_name

    # print(f"Renaming dir {day_dir} to {new_day_dir}")
    day_dir.rename(new_day_dir)


def main(input_dir: str = "/home/timur_akht/egowalk_raw_mnt"):
    input_dir = Path(input_dir)
    day_dirs = sorted(input_dir.glob("*/"))
    for day_dir in day_dirs:
        process_day_dir(day_dir)


if __name__ == "__main__":
    fire.Fire(main)
