preprocess:
	python3 src/segmentation.py sample_frames --input_path /mnt/data/2024_07_02__14_05_34 --output_path /mnt/data/preprocess_data

process:
	python3 src/segmentation.py generate --input_path /mnt/data/preprocess_data/ --criterion area --weights_path /mnt/code/weights/sam_vit.pth

remove:
	rm -rf /mnt/data/preprocess_data
	rm -rf /mnt/data/preprocess_data/masks_area
