preprocess:
	python3 src/segmentation.py sample_frames --input_path /mnt/data/2024_07_02__14_05_34 --output_path /mnt/data/preprocess_data

process:
	python3 src/segmentation.py generate --input_path /mnt/data/preprocess_data/ --criterion area --weights_path /mnt/code/weights/sam_vit.pth 

postprocess:
	python3 src/segmentation.py apply_masks --images_path /mnt/data/preprocess_data/frames/ --masks_path /mnt/data/preprocess_data/masks_area/ --output_path /mnt/data/output
	
remove:
	rm -rf /mnt/data/preprocess_data
	rm -rf /mnt/data/preprocess_data/masks_area
	rm -rf /mnt/data/output