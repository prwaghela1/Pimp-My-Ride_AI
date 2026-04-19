import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a boolean or uint8 mask to an L-mode PIL image."""
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def extract_canny_edges(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Run Canny edge detection and return an RGB PIL image for ControlNet."""
    gray = cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(edges_rgb)


def largest_mask(masks: list[dict]) -> np.ndarray:
    """Return the boolean mask with the largest area from SAM output."""
    return max(masks, key=lambda m: m["area"])["segmentation"]


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Slightly expand the mask so inpainting covers hard edges cleanly."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def resize_to_multiple(image: Image.Image, multiple: int = 8) -> Image.Image:
    """Resize image so both dimensions are multiples of `multiple`."""
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def blend_images(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    feather: int = 20,
) -> Image.Image:
    """Alpha-composite generated pixels onto original using a feathered mask."""
    orig_np = np.array(original.convert("RGBA")).astype(float)
    gen_np = np.array(generated.convert("RGBA")).astype(float)
    mask_np = np.array(mask.convert("L")).astype(float) / 255.0

    if feather > 0:
        mask_np = cv2.GaussianBlur(mask_np, (feather * 2 + 1, feather * 2 + 1), 0)

    alpha = mask_np[..., np.newaxis]
    blended = gen_np * alpha + orig_np * (1 - alpha)
    return Image.fromarray(blended.astype(np.uint8), "RGBA").convert("RGB")
