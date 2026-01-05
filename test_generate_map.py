import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np
import os
from pathlib import Path

# Import the functions to test
from generate_map import download_tile, get_equirectangular_coords, get_winkel_tripel_coords, reproject_and_save, get_mercator_canvas

class TestMapGenerator(unittest.TestCase):

    @patch('generate_map.download_tile_content')
    def test_download_tile_success(self, mock_content):
        img = Image.new('RGB', (256, 256), color='red')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        mock_content.return_value = img_byte_arr.getvalue()

        result = download_tile("http://example.com/tile.png")
        self.assertIsNotNone(result)
        self.assertEqual(result.size, (256, 256))

    def test_get_equirectangular_coords(self):
        # Just check it returns correctly shaped float32 arrays
        iy, ix, mask = get_equirectangular_coords(256, 128, 0, 64, 512, 512)
        self.assertEqual(iy.shape, (64, 256))
        self.assertEqual(iy.dtype, np.float32)
        self.assertEqual(mask.shape, (64, 256))

    def test_get_winkel_tripel_coords(self):
        iy, ix, mask = get_winkel_tripel_coords(256, 156, 0, 40, 512, 512)
        self.assertEqual(iy.shape, (40, 256))
        self.assertEqual(iy.dtype, np.float32)

    @patch('generate_map.download_tile')
    def test_get_mercator_canvas(self, mock_download):
        mock_download.return_value = Image.new('RGB', (256, 256), color='white')
        canvas = get_mercator_canvas(0, "esri")
        self.assertEqual(canvas.size, (256, 256))

    @patch('PIL.Image.Image.save')
    def test_reproject_and_save(self, mock_save):
        input_img = Image.new('RGB', (512, 512), color='blue')
        # Test mercator
        reproject_and_save(input_img, "mercator", 1.0, "out.png", 256)
        mock_save.assert_called_with("out.png")
        
        # Test equirectangular
        reproject_and_save(input_img, "equirectangular", 1.0, "out_equi.png", 128)
        self.assertTrue(mock_save.called)

if __name__ == '__main__':
    unittest.main()