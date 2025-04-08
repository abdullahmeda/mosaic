import io
import base64

from PIL import Image


def base64_encode_image(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image


def image_decode_base64(base64_string: str,) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))


def base64_encode_image_list(image_list: list[Image.Image], format: str = "PNG") -> list[str]:
    return [base64_encode_image(image, format) for image in image_list]


def image_decode_base64_list(base64_string_list: list[str]) -> list[Image.Image]:
    return [image_decode_base64(base64_string) for base64_string in base64_string_list]


def resize_image(image: Image.Image, max_image_width, max_image_height):
    """ Resize an image to fit within the given width and height """

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
    return [resize_image(image, max_image_width, max_image_height) for image in image_list]