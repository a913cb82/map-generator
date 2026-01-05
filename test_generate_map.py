import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np
import math
import os

# Import the functions to test
from generate_map import download_tile, reproject_to_equirectangular, reproject_to_winkel_tripel, generate_map

class TestMapGenerator(unittest.TestCase):

    @patch('generate_map.download_tile_content')
    def test_download_tile_success(self, mock_content):
        # Create a small dummy image
        img = Image.new('RGB', (256, 256), color='red')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Mock cached content
        mock_content.return_value = img_byte_arr

        result = download_tile("http://example.com/tile.png")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.size, (256, 256))
        mock_content.assert_called_once_with("http://example.com/tile.png")

    @patch('generate_map.download_tile_content')
    def test_download_tile_failure(self, mock_content):
        # Mock failure (e.g. 404 or other error returning None)
        mock_content.return_value = None
        
        result = download_tile("http://example.com/tile.png")
        self.assertIsNone(result)

    def test_reproject_to_equirectangular(self):
        # Create a dummy Mercator-like image (square)
        w, h = 256, 256
        img = Image.new('RGB', (w, h), color='blue')
        
        result = reproject_to_equirectangular(img)
        
        # Equirectangular should be 2:1 aspect ratio
        self.assertEqual(result.size, (w, w // 2))
        self.assertIsInstance(result, Image.Image)

    def test_reproject_to_winkel_tripel(self):
        # Create a dummy Mercator-like image (square)
        w, h = 256, 256
        img = Image.new('RGB', (w, h), color='green')
        
        result = reproject_to_winkel_tripel(img)
        
        # Winkel Tripel should have roughly 1.636 aspect ratio
        self.assertEqual(result.size[0], w)
        self.assertEqual(result.size[1], int(w / 1.636))
        self.assertIsInstance(result, Image.Image)

    @patch('generate_map.download_tile')
    @patch('PIL.Image.Image.save')
    def test_generate_map_loop(self, mock_save, mock_download):
        # Mock download_tile to return a small image
        dummy_tile = Image.new('RGB', (256, 256), color='white')
        mock_download.return_value = dummy_tile
        
        # Test zoom 1 (2x2 tiles)
        generate_map(zoom=1, map_type="esri", projection="mercator", output_path="test.png")
        
        # Should have tried to download 4 tiles
        self.assertEqual(mock_download.call_count, 4)
        mock_save.assert_called_once_with("test.png")

    @patch('generate_map.download_tile')
    @patch('generate_map.reproject_to_winkel_tripel')
    @patch('PIL.Image.Image.save')
    def test_generate_map_with_projection(self, mock_save, mock_reproject, mock_download):
        dummy_tile = Image.new('RGB', (256, 256), color='white')
        mock_download.return_value = dummy_tile
        mock_reproject.return_value = Image.new('RGB', (100, 50))
        
        generate_map(zoom=0, map_type="esri", projection="winkel_tripel", output_path="test.png")
        
        mock_reproject.assert_called_once()
        mock_save.assert_called_once_with("test.png")

if __name__ == '__main__':
    unittest.main()
