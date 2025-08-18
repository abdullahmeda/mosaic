"""Common utility functions and logging configuration for mosaic"""

import io
import base64

from PIL import Image

import logging
import sys
from typing import Optional

# Main application logger name
LOGGER_NAME = "mosaic"

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
    """Setup centralized logging configuration for mosaic.
    
    This should be called once at application startup to configure the main logger.
    All other modules should use get_logger() to get child loggers.
    
    Parameters:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string for log messages
        
    Returns:
        The configured main logger
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Default format if none provided
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create main logger
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    _logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    _logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    _logger.propagate = False
    
    return _logger


def base64_encode_image(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image


def image_decode_base64(
    base64_string: str,
) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))


def base64_encode_image_list(
    image_list: list[Image.Image], format: str = "PNG"
) -> list[str]:
    return [base64_encode_image(image, format) for image in image_list]


def image_decode_base64_list(base64_string_list: list[str]) -> list[Image.Image]:
    return [image_decode_base64(base64_string) for base64_string in base64_string_list]


def resize_image(image: Image.Image, max_image_width, max_image_height):
    """Resize an image to fit within the given width and height"""

    img_width, img_height = image.size
    aspect_ratio = img_width / img_height

    if img_width > max_image_width:
        new_width = max_image_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = img_width
        new_height = img_height

    if new_height > max_image_height:
        new_height = max_image_height
        new_width = int(new_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    return image


def resize_image_list(image_list: list[Image.Image], max_image_width, max_image_height):
    return [
        resize_image(image, max_image_width, max_image_height) for image in image_list
    ]
