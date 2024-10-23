# cython: boundscheck=False, wraparound=False
import os
import cv2
import numpy as np
cimport numpy as cnp  # cimport для numpy

def _apply_masks(str images_path, str masks_path, str output_path):
    """
    Apply masks to images.
    
    Args:
        images_path (str): Path to the input images.
        masks_path (str): Path to the masks.
        output_path (str): Path to save the masked images.
        
    Returns:
        None
    """
    cdef list images_id = os.listdir(images_path)
    cdef list masks_id = os.listdir(masks_path)
    
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    cdef str image_path, mask_path, output_file_path
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image  # 3D-массив для цветного изображения
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] mask   # 2D-массив для маски
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] masked_image  # 3D-массив для результата
    cdef int image_height, image_width
    cdef int mask_height, mask_width
    
    for image_id, mask_id in zip(images_id, masks_id):
        # Construct full paths
        image_path = os.path.join(images_path, image_id)
        mask_path = os.path.join(masks_path, mask_id)
        output_file_path = os.path.join(output_path, image_id)

        # Load the image and mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Error: Unable to load {image_path} or {mask_path}")
            continue

        # Получаем размеры изображения и маски
        image_height, image_width = image.shape[0], image.shape[1]
        mask_height, mask_width = mask.shape[0], mask.shape[1]

        # Resize the mask to match image dimensions if necessary
        if mask_height != image_height or mask_width != image_width:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_AREA)

        # Apply the mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Save the resulting masked image
        cv2.imwrite(output_file_path, masked_image)

        print(f"Saved masked image to {output_file_path}")