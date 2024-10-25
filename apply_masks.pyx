# apply_masks.pyx

# cython: boundscheck=False
# cython: wraparound=False

import os
import cv2
import numpy as np
cimport numpy as np  # Import Cython's numpy

def apply_masks(str images_path, str masks_path, str output_path):
    """
    Apply masks to images.

    Args:
        images_path (str): Path to the input images.
        masks_path (str): Path to the masks.
        output_path (str): Path to save the masked images.

    Returns:
        None
    """
    cdef list images_id
    cdef str image_id, image_path, mask_path, mask_filename, output_image_path
    cdef np.ndarray[np.uint8_t, ndim=3] image
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    cdef np.ndarray[np.uint8_t, ndim=3] masked_image
    cdef int image_height, image_width
    cdef int mask_height, mask_width

    # Get list of image filenames
    images_id = sorted(os.listdir(images_path))

    # Check if output directory exists, create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_id in images_id:
        # Construct the paths
        image_path = os.path.join(images_path, image_id)
        mask_filename = os.path.splitext(image_id)[0] + '.png'  # Assuming masks are .png files
        mask_path = os.path.join(masks_path, mask_filename)
        output_image_path = os.path.join(output_path, image_id)

        # Check if the corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Mask for image {image_id} not found, skipping.")
            continue

        # Load the image and mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading image or mask for {image_id}, skipping.")
            continue

        # Ensure the mask is of the same size as the image
        image_height, image_width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]
        if (mask_height != image_height) or (mask_width != image_width):
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_AREA)

        # Create an empty output image
        masked_image = np.zeros_like(image)

        # Apply the mask
        apply_mask_to_image(image, mask, masked_image)

        # Save the result
        cv2.imwrite(output_image_path, masked_image)

cdef void apply_mask_to_image(np.ndarray[np.uint8_t, ndim=3] image,
                              np.ndarray[np.uint8_t, ndim=2] mask,
                              np.ndarray[np.uint8_t, ndim=3] output_image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef int c = image.shape[2]
    cdef int i, j, k
    cdef np.uint8_t m

    for i in range(h):
        for j in range(w):
            m = mask[i, j]
            if m == 0:
                for k in range(c):
                    output_image[i, j, k] = 0
            else:
                for k in range(c):
                    output_image[i, j, k] = image[i, j, k]