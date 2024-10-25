import unittest
import os
import shutil
import tempfile
import numpy as np
import cv2

# Import the compiled Cython module
from src.apply_masks import apply_masks

class TestApplyMasks(unittest.TestCase):

    def setUp(self):
        # Create temporary directories for images, masks, and output
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, 'images')
        self.masks_dir = os.path.join(self.test_dir, 'masks')
        self.output_dir = os.path.join(self.test_dir, 'output')

        os.makedirs(self.images_dir)
        os.makedirs(self.masks_dir)

    def tearDown(self):
        # Remove the directory 
        shutil.rmtree(self.test_dir)

    def test_empty_images_directory(self):
        """Test processing when the images directory is empty."""
        # images_dir is already empty
        apply_masks(self.images_dir, self.masks_dir, self.output_dir)
        
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

    def test_missing_mask_for_image(self):
        """Test handling when a mask is missing for an image."""
        image_filename = 'test_image.jpg'

        image_path = os.path.join(self.images_dir, image_filename)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_image)

        apply_masks(self.images_dir, self.masks_dir, self.output_dir)

        self.assertTrue(os.path.exists(self.output_dir))
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

    def test_successful_mask_application(self):
        """Test successful application of a mask to an image."""
        image_filename = 'test_image.jpg'
        image_path = os.path.join(self.images_dir, image_filename)
        dummy_image = np.full((100, 100, 3), 255, dtype=np.uint8)  
        cv2.imwrite(image_path, dummy_image)

        mask_filename = 'test_image.png' 
        mask_path = os.path.join(self.masks_dir, mask_filename)
        dummy_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(dummy_mask, (50, 50), 25, (255), -1) 
        cv2.imwrite(mask_path, dummy_mask)

        apply_masks(self.images_dir, self.masks_dir, self.output_dir)

        output_image_path = os.path.join(self.output_dir, image_filename)
        self.assertTrue(os.path.exists(output_image_path))

        output_image = cv2.imread(output_image_path)
        self.assertTrue((output_image[50, 50] == [255, 255, 255]).all())
        self.assertTrue((output_image[10, 10] == [0, 0, 0]).all())

    def test_empty_images_and_masks_directories(self):
        """Test processing when both images and masks directories are empty."""
        # Both images_dir and masks_dir are empty
        apply_masks(self.images_dir, self.masks_dir, self.output_dir)
        # Check that output directory is created but remains empty
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

if __name__ == '__main__':
    unittest.main()