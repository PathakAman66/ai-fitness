"""
Unit tests for enhanced image validation in ImageProcessor
"""
import pytest
import numpy as np
from backend.utils.image_processor import ImageProcessor


class TestImageValidation:
    """Test suite for enhanced image validation"""

    def test_decode_base64_empty_string(self):
        """Test that empty base64 string raises ValueError"""
        with pytest.raises(ValueError, match="Base64 string must be a non-empty string"):
            ImageProcessor.decode_base64_image("")

    def test_decode_base64_none_value(self):
        """Test that None base64 string raises ValueError"""
        with pytest.raises(ValueError, match="Base64 string must be a non-empty string"):
            ImageProcessor.decode_base64_image(None)

    def test_decode_base64_invalid_format(self):
        """Test that invalid base64 format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid base64 format"):
            ImageProcessor.decode_base64_image("invalid@base64!")

    def test_validate_image_none(self):
        """Test that None image raises ValueError"""
        with pytest.raises(ValueError, match="Image is None"):
            ImageProcessor.validate_image(None)

    def test_validate_image_empty(self):
        """Test that empty image raises ValueError"""
        empty_image = np.array([])
        with pytest.raises(ValueError, match="Image is empty"):
            ImageProcessor.validate_image(empty_image)

    def test_validate_image_too_small(self):
        """Test that image smaller than minimum size raises ValueError"""
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image dimensions .* are too small"):
            ImageProcessor.validate_image(small_image)

    def test_validate_image_too_large(self):
        """Test that image larger than maximum size raises ValueError"""
        large_image = np.zeros((5000, 5000, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image dimensions .* are too large"):
            ImageProcessor.validate_image(large_image)

    def test_validate_image_invalid_channels(self):
        """Test that image with invalid number of channels raises ValueError"""
        invalid_image = np.zeros((200, 200, 5), dtype=np.uint8)  # 5 channels
        with pytest.raises(ValueError, match="Image has .* channels"):
            ImageProcessor.validate_image(invalid_image)

    def test_validate_image_valid(self):
        """Test that valid image passes validation"""
        valid_image = np.zeros((200, 200, 3), dtype=np.uint8)
        assert ImageProcessor.validate_image(valid_image) is True
