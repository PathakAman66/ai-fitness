"""
ImageProcessor utility class for FastAPI Fitness Backend
Handles image encoding/decoding and format conversion for pose detection
"""
import base64
import numpy as np
import cv2
from typing import Optional
from fastapi import UploadFile
import io
from PIL import Image


class ImageProcessor:
    """
    Utility class for processing images in various formats
    Supports base64 encoding/decoding and multipart file uploads
    """
    
    # Maximum image size in bytes (10MB)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> np.ndarray:
        """
        Decode a base64 encoded image string to numpy array
        
        Args:
            base64_string: Base64 encoded image string (with or without data URI prefix)
            
        Returns:
            numpy.ndarray: Decoded image in BGR format (OpenCV format)
            
        Raises:
            ValueError: If the base64 string is invalid or cannot be decoded
        """
        if not base64_string or not isinstance(base64_string, str):
            raise ValueError("Base64 string must be a non-empty string")
        
        try:
            # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
            if ',' in base64_string:
                base64_string = base64_string.split(',', 1)[1]
            
            # Remove any whitespace
            base64_string = base64_string.strip()
            
            if not base64_string:
                raise ValueError("Base64 string is empty after removing data URI prefix")
            
            # Validate base64 format (basic check)
            import re
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_string):
                raise ValueError("Invalid base64 format")
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            if len(image_bytes) == 0:
                raise ValueError("Decoded base64 data is empty")
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image using OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image - invalid image data or unsupported format")
            
            return image
            
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format: str = '.jpg', quality: int = 95) -> str:
        """
        Encode a numpy array image to base64 string
        
        Args:
            image: Image as numpy array in BGR format (OpenCV format)
            format: Output image format ('.jpg', '.png', etc.)
            quality: JPEG quality (0-100), only applies to JPEG format
            
        Returns:
            str: Base64 encoded image string (without data URI prefix)
            
        Raises:
            ValueError: If the image cannot be encoded
        """
        try:
            # Validate image
            if image is None or image.size == 0:
                raise ValueError("Image is empty or None")
            
            # Set encoding parameters based on format
            if format.lower() in ['.jpg', '.jpeg']:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format.lower() == '.png':
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                encode_params = []
            
            # Encode image to bytes
            success, encoded_image = cv2.imencode(format, image, encode_params)
            
            if not success:
                raise ValueError(f"Failed to encode image to {format} format")
            
            # Convert to base64
            base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {str(e)}")
    
    @staticmethod
    async def process_uploaded_file(file: UploadFile) -> np.ndarray:
        """
        Process an uploaded file from multipart form data
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            numpy.ndarray: Decoded image in BGR format (OpenCV format)
            
        Raises:
            ValueError: If the file is invalid, too large, or unsupported format
        """
        if not file:
            raise ValueError("No file provided")
        
        if not file.filename:
            raise ValueError("File must have a filename")
        
        try:
            # Check file size
            contents = await file.read()
            file_size = len(contents)
            
            if file_size == 0:
                raise ValueError("Uploaded file is empty")
            
            if file_size > ImageProcessor.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"File size ({file_size} bytes) exceeds maximum allowed size "
                    f"({ImageProcessor.MAX_IMAGE_SIZE} bytes)"
                )
            
            # Check file extension
            if file.filename:
                file_ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
                if file_ext and file_ext not in ImageProcessor.SUPPORTED_FORMATS:
                    raise ValueError(
                        f"Unsupported file format: {file_ext}. "
                        f"Supported formats: {', '.join(ImageProcessor.SUPPORTED_FORMATS)}"
                    )
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(contents, np.uint8)
            
            # Decode image using OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode uploaded image - invalid image data or unsupported format")
            
            return image
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to process uploaded file: {str(e)}")
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate that an image is suitable for pose detection
        
        Args:
            image: Image as numpy array
            
        Returns:
            bool: True if image is valid, False otherwise
            
        Raises:
            ValueError: If image is invalid with descriptive error message
        """
        try:
            # Check if image is None
            if image is None:
                raise ValueError("Image is None")
            
            # Check if image is empty
            if image.size == 0:
                raise ValueError("Image is empty")
            
            # Check image dimensions
            if len(image.shape) < 2:
                raise ValueError("Image must have at least 2 dimensions")
            
            height, width = image.shape[:2]
            
            # Check minimum dimensions (at least 100x100 pixels)
            if height < 100 or width < 100:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) are too small. "
                    f"Minimum required: 100x100 pixels"
                )
            
            # Check maximum dimensions (prevent memory issues)
            max_dimension = 4096
            if height > max_dimension or width > max_dimension:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) are too large. "
                    f"Maximum allowed: {max_dimension}x{max_dimension} pixels"
                )
            
            # Check if image has valid color channels (1 for grayscale, 3 for BGR/RGB)
            if len(image.shape) == 3:
                channels = image.shape[2]
                if channels not in [1, 3, 4]:
                    raise ValueError(
                        f"Image has {channels} channels. Expected 1 (grayscale), 3 (BGR/RGB), or 4 (BGRA/RGBA)"
                    )
            
            # Check data type
            if image.dtype not in [np.uint8, np.float32, np.float64]:
                raise ValueError(f"Unsupported image data type: {image.dtype}")
            
            return True
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Image validation failed: {str(e)}")
    
    @staticmethod
    def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Image as numpy array
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            
        Returns:
            numpy.ndarray: Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height, 1.0)
        
        # Only resize if image is larger than max dimensions
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
