#!/usr/bin/env python3
"""
Minimal Gradio demo for a locally-trained LoRA checkpoint
on Black-Forest-Labs’ FLUX pipeline.
"""

import torch, gradio as gr
from diffusers import FluxPipeline

# ── Paths ──────────────────────────────────────────────────────────────
BASE_MODEL_ID   = "black-forest-labs/FLUX.1-dev"
LORA_MODEL_ID   = "/home/raluca/2D-multiview-generation/models/4views.safetensors"

# ── Pipeline setup ─────────────────────────────────────────────────────


pipe = FluxPipeline.from_pretrained(
    BASE_MODEL_ID,
    #device_map="balanced",  # "balanced" keeps the UNet on GPU-0 and off-loads text-encoder + VAE to CPURAM, so the 4090’s 24 GB are enough.
    torch_dtype=torch.bfloat16,  # use float16 for faster inference
)

# pipe.reset_device_map()  # reset the device map to the default one, so we can load LoRA weights
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# If you trained the LoRA with the Diffusers LoRA utility, `rank` is auto-detected.
print("Loading LoRA weights …")
pipe.load_lora_weights(LORA_MODEL_ID)

# Fuse for a small speed bump (you can comment this out to keep LoRA editable)
pipe.fuse_lora()
pipe.set_progress_bar_config(disable=False)   # show tqdm



# ── Inference fn ───────────────────────────────────────────────────────
def infer(prompt, steps=10, guidance=4.0, seed=0):
    g = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
    if "[FOUR-VIEWS]" not in prompt:
        prompt = f"[FOUR-VIEWS] {prompt}"
    image = pipe(prompt,
                 num_inference_steps=steps,
                 guidance_scale=guidance,
                 height=512, width=512,
                 generator=g).images[0]
    return image

# ── Gradio UI ──────────────────────────────────────────────────────────
demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Textbox(label="Prompt",
                       value="[FOUR-VIEWS] a red desk lamp from multiple views;[TOP-LEFT] This photo shows a 45-degree angle of desk lamp;[TOP-RIGHT] This photo shows a high-angle shot of the lamp; [BOTTOM-LEFT] Here is a side view shot of lamp; [BOTTOM-RIGHT] The back view of the desk lamp.",
                       lines=2),
            gr.Slider(4, 50, value=5, step=1, label="Inference steps"),
            gr.Slider(0, 15, value=4, step=0.5, label="Guidance scale"),
        ],
        outputs=gr.Image(type="pil", label="Result"),
        title="Four-View Grid LoRA",
        allow_flagging="never")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)