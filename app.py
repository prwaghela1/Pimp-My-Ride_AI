"""
Pimp My Ride AI — Gradio interface
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import gradio as gr
from PIL import Image

from pipeline import restyle_car, SAM_CHECKPOINT_URL, SAM_MODEL_TYPE

# ---------------------------------------------------------------------------
# SAM checkpoint management
# ---------------------------------------------------------------------------
SAM_CHECKPOINT_DIR = Path("checkpoints")
SAM_CHECKPOINT_PATH = SAM_CHECKPOINT_DIR / "sam_vit_h_4b8939.pth"


def _ensure_sam_checkpoint() -> str:
    SAM_CHECKPOINT_DIR.mkdir(exist_ok=True)
    if not SAM_CHECKPOINT_PATH.exists():
        print(f"Downloading SAM checkpoint (~2.4 GB) → {SAM_CHECKPOINT_PATH} …")
        urllib.request.urlretrieve(
            SAM_CHECKPOINT_URL,
            SAM_CHECKPOINT_PATH,
            reporthook=lambda b, bs, t: print(
                f"\r  {min(b * bs, t) / 1e9:.2f} / {t / 1e9:.2f} GB", end=""
            ),
        )
        print("\nDownload complete.")
    return str(SAM_CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# Preset prompts
# ---------------------------------------------------------------------------
PRESETS = {
    "Racing Red":      "matte racing red sports car with carbon fiber accents, aggressive styling, photorealistic",
    "Cyberpunk Neon":  "cyberpunk neon glowing car with electric blue and purple LED underglow, futuristic city background",
    "Military Camo":   "military camouflage car, olive green and tan camo paint, rugged off-road look, photorealistic",
    "Pearlescent Gold":"pearlescent gold luxury car, glossy metallic paint, chrome trim, showroom lighting",
    "Matte Black":     "matte black stealth car, murdered-out look, black rims, smoked windows, photorealistic",
}

# ---------------------------------------------------------------------------
# Core inference wrapper
# ---------------------------------------------------------------------------

def run_pipeline(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_steps: int,
    canny_low: int,
    canny_high: int,
    seed: int,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    if image is None:
        raise gr.Error("Please upload a car image first.")
    if not prompt.strip():
        raise gr.Error("Please enter a style prompt.")

    sam_path = _ensure_sam_checkpoint()
    seed_val = int(seed) if seed >= 0 else None

    outputs = restyle_car(
        original_image=image,
        prompt=prompt,
        sam_checkpoint=sam_path,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        canny_low=canny_low,
        canny_high=canny_high,
        seed=seed_val,
    )
    return (
        outputs["result"],
        outputs["segmented"],
        outputs["mask"],
        outputs["canny"],
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Pimp My Ride AI",
    theme=gr.themes.Base(
        primary_hue="rose",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        #title { text-align: center; font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
        #subtitle { text-align: center; color: #888; margin-top: 4px; margin-bottom: 24px; }
        .generate-btn { background: linear-gradient(135deg, #e11d48, #be123c) !important; color: white !important; }
    """,
) as demo:

    gr.Markdown("# Pimp My Ride AI", elem_id="title")
    gr.Markdown(
        "Upload a car photo, describe your dream style, and let AI restyle it — "
        "powered by **SAM + ControlNet + Stable Diffusion**.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ---- Left column: inputs ----------------------------------------
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Car Image",
                type="pil",
                sources=["upload", "clipboard"],
                height=320,
            )

            with gr.Group():
                gr.Markdown("### Style Preset")
                preset_dropdown = gr.Dropdown(
                    choices=["(none)"] + list(PRESETS.keys()),
                    value="(none)",
                    label="Quick Presets",
                    interactive=True,
                )

            with gr.Group():
                gr.Markdown("### Prompt")
                prompt_box = gr.Textbox(
                    label="Style Prompt",
                    placeholder="e.g. matte black sports car with red racing stripes, photorealistic",
                    lines=3,
                )
                neg_prompt_box = gr.Textbox(
                    label="Negative Prompt",
                    value="ugly, low quality, blurry, deformed, watermark, text, distorted wheels",
                    lines=2,
                )

            with gr.Accordion("Advanced Settings", open=False):
                strength_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.75,
                    step=0.05,
                    label="Style Strength",
                    info="Higher = more stylised, lower = closer to original",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale",
                    info="How strictly the model follows the prompt",
                )
                steps_slider = gr.Slider(
                    minimum=10,
                    maximum=60,
                    value=30,
                    step=5,
                    label="Inference Steps",
                )
                with gr.Row():
                    canny_low = gr.Slider(
                        minimum=0, maximum=255, value=100, step=5,
                        label="Canny Low Threshold",
                    )
                    canny_high = gr.Slider(
                        minimum=0, maximum=255, value=200, step=5,
                        label="Canny High Threshold",
                    )
                seed_box = gr.Number(
                    value=-1,
                    label="Seed (-1 = random)",
                    precision=0,
                )

            generate_btn = gr.Button(
                "Pimp My Ride!", variant="primary", elem_classes=["generate-btn"]
            )

        # ---- Right column: outputs --------------------------------------
        with gr.Column(scale=1):
            result_image = gr.Image(label="Restyled Car", height=320, interactive=False)

            with gr.Row():
                segmented_image = gr.Image(
                    label="SAM Segmentation", height=200, interactive=False
                )
                mask_image = gr.Image(
                    label="Inpainting Mask", height=200, interactive=False
                )
                canny_image = gr.Image(
                    label="ControlNet Edges", height=200, interactive=False
                )

    # ---- Examples -------------------------------------------------------
    gr.Markdown("### Example Prompts")
    gr.Examples(
        examples=[
            ["(none)", "matte black sports car with gold racing stripes, dramatic lighting"],
            ["Cyberpunk Neon", ""],
            ["Racing Red", ""],
            ["Pearlescent Gold", ""],
        ],
        inputs=[preset_dropdown, prompt_box],
        label="Click to load",
    )

    # ---- Event handlers -------------------------------------------------
    def apply_preset(preset_name: str, current_prompt: str) -> str:
        if preset_name == "(none)":
            return current_prompt
        return PRESETS[preset_name]

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown, prompt_box],
        outputs=[prompt_box],
    )

    generate_btn.click(
        fn=run_pipeline,
        inputs=[
            input_image,
            prompt_box,
            neg_prompt_box,
            strength_slider,
            guidance_slider,
            steps_slider,
            canny_low,
            canny_high,
            seed_box,
        ],
        outputs=[result_image, segmented_image, mask_image, canny_image],
    )

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share="--share" in sys.argv,
        show_error=True,
    )
