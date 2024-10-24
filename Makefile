preprocess:
	python3 src/segmentation.py sample_frames --input_path /mnt/data/2024_07_02__14_05_34 --output_path /mnt/data/preprocess_data

process:
	


remove:
	rm -rf /mnt/data/preprocess_data
