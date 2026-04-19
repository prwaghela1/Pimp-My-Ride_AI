"""
Core inference pipeline:
  1. SAM  — segment the car from background
  2. Canny — extract structural edges from the segmented car
  3. ControlNet + SD Inpainting — generate the restyled car
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

from utils import (
    extract_canny_edges,
    largest_mask,
    dilate_mask,
    mask_to_pil,
    resize_to_multiple,
    blend_images,
)


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
SAM_MODEL_TYPE = "vit_h"

SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

# ---------------------------------------------------------------------------
# Global singletons (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_sam: SamAutomaticMaskGenerator | None = None
_pipe: StableDiffusionControlNetInpaintPipeline | None = None
_device: str | None = None


def _get_device() -> str:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def _load_sam(checkpoint_path: str) -> SamAutomaticMaskGenerator:
    global _sam
    if _sam is None:
        device = _get_device()
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint_path)
        sam.to(device)
        _sam = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
        )
    return _sam


def _load_pipe() -> StableDiffusionControlNetInpaintPipeline:
    global _pipe
    if _pipe is None:
        device = _get_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL, torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()

        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()

        pipe.to(device)
        _pipe = pipe
    return _pipe


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_car(
    image: Image.Image,
    sam_checkpoint: str,
) -> tuple[np.ndarray, Image.Image]:
    """
    Returns (boolean_mask, masked_car_PIL_image).
    The mask covers the dominant (largest) object — assumed to be the car.
    """
    sam_gen = _load_sam(sam_checkpoint)
    image_np = np.array(image.convert("RGB"))
    masks = sam_gen.generate(image_np)

    if not masks:
        raise ValueError("SAM found no objects in the image.")

    mask = largest_mask(masks)            # shape (H, W) bool
    mask = dilate_mask(mask, kernel_size=11)

    # Visualise: car pixels visible, background black
    masked_np = image_np.copy()
    masked_np[~mask] = 0
    return mask, Image.fromarray(masked_np)


def restyle_car(
    original_image: Image.Image,
    prompt: str,
    sam_checkpoint: str,
    negative_prompt: str = (
        "ugly, low quality, blurry, deformed, watermark, text, distorted wheels"
    ),
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    canny_low: int = 100,
    canny_high: int = 200,
    seed: int | None = None,
) -> dict[str, Image.Image]:
    """
    Full pipeline: SAM → Canny → ControlNet inpainting.

    Returns a dict with keys:
        "result"      — final composited image
        "mask"        — inpainting mask (white = car)
        "canny"       — Canny edge map used for ControlNet
        "segmented"   — car cutout on black background
    """
    original_image = resize_to_multiple(original_image.convert("RGB"))
    w, h = original_image.size

    # --- Step 1: SAM segmentation ----------------------------------------
    bool_mask, segmented = segment_car(original_image, sam_checkpoint)
    mask_pil = mask_to_pil(bool_mask)                        # L-mode, 0/255

    # --- Step 2: Canny edges (on the car region only) --------------------
    canny_image = extract_canny_edges(segmented, canny_low, canny_high)
    canny_image = canny_image.resize((w, h))

    # --- Step 3: SD ControlNet Inpainting --------------------------------
    pipe = _load_pipe()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_get_device()).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_image,
        mask_image=mask_pil,
        control_image=canny_image,
        width=w,
        height=h,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    # Composite: generated car on original background using feathered mask
    composited = blend_images(original_image, result, mask_pil, feather=15)

    return {
        "result": composited,
        "mask": mask_pil.convert("RGB"),
        "canny": canny_image,
        "segmented": segmented,
    }
