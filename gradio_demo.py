# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

import gradio as gr
from torch import Generator
import torch
from tstitch_sd_utils import TStitchSD, TStitchSDXL, get_tstitch_pipepline_sdxl_lcm_lora
import numpy as np
from diffusers import UNet2DConditionModel, LCMScheduler, AutoencoderTiny

all_unet = {}
cache_dir = './pretrained_models'


def get_unet(model_name):
    if model_name in all_unet:
        return all_unet[model_name]

    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float16,
                                                cache_dir=cache_dir, fp16=True)
    all_unet[model_name] = unet
    return unet


def get_tstitch_sd_14_pipepline(large_sd, small_sd):
    small_unet = get_unet(small_sd)

    keys = list(all_unet.keys())
    for key in keys:
        if key != small_sd and key != large_sd:
            del all_unet[key]

    torch.cuda.empty_cache()

    pipe_sd = TStitchSD.from_pretrained(large_sd, torch_dtype=torch.float16, safety_checker=None,
                                        requires_safety_checker=False, cache_dir=cache_dir)
    pipe_sd.unet = torch.nn.ModuleList([small_unet, pipe_sd.unet])
    pipe_sd.unet.dtype = small_unet.dtype
    pipe_sd = pipe_sd.to('cuda')
    return pipe_sd


def get_tstitch_sdxl_pipepline(large_sd, small_sd):
    small_unet = get_unet(small_sd)
    large_unet = get_unet(large_sd)

    keys = list(all_unet.keys())
    for key in keys:
        if key != small_sd and key != large_sd:
            del all_unet[key]

    torch.cuda.empty_cache()
    pipe_sd = TStitchSDXL.from_pretrained(large_sd, unet=large_unet, torch_dtype=torch.float16, variant="fp16",
                                          safety_checker=None, requires_safety_checker=False, cache_dir=cache_dir)
    pipe_sd.unet = torch.nn.ModuleList([small_unet, large_unet])
    pipe_sd.unet.dtype = small_unet.dtype
    pipe_sd = pipe_sd.to('cuda')
    return pipe_sd

def get_stitch_sdxl_lcm_pipeline(small_sd, large_sd):
    lcm_pipe = get_tstitch_pipepline_sdxl_lcm_lora(large_sd, small_sd)
    lcm_pipe.scheduler = LCMScheduler.from_config(lcm_pipe.scheduler.config)
    lcm_pipe.to("cuda")
    lcm_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16,
                                                   use_safetensors=True, cache_dir=cache_dir).to("cuda")
    return lcm_pipe

def sd_14_15_func(prompt, ratios, small_sd, large_sd, guidance_scale, num_inference_steps, seed,
                  progress=gr.Progress()):
    torch.cuda.empty_cache()
    progress(0, desc="Loading weights...")
    pipe_sd = get_tstitch_sd_14_pipepline(large_sd, small_sd)
    outputs = []
    generator = torch.manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
    for r in progress.tqdm(ratios, desc="Generating images"):
        image = pipe_sd(prompt, unet_s_ratio=r, num_inference_steps=num_inference_steps, latents=latents, height=512, width=512,
                guidance_scale=guidance_scale, generator=generator).images[0]
        outputs.append((image, f'ratio = {r}'))
    return outputs


def sdxl_func(prompt, negative_prompts, ratios, small_sd, large_sd, guidance_scale, num_inference_steps, seed,
              progress=gr.Progress()):
    progress(0, desc="Loading weights...")
    torch.cuda.empty_cache()
    pipe_sd = get_tstitch_sdxl_pipepline(large_sd, small_sd)
    outputs = []
    generator = torch.manual_seed(seed)
    latents = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
    for r in progress.tqdm(ratios, desc="Generating images"):
        image = pipe_sd(prompt, negative_prompts=negative_prompts, unet_s_ratio=r, num_inference_steps=num_inference_steps,
                 latents=latents, height=512, width=512, guidance_scale=guidance_scale, generator=generator).images[0]
        outputs.append((image, f'ratio = {r}'))
    return outputs


def sdxl_lcm_func(prompt, negative_prompts, r, small_sd, large_sd, guidance_scale, num_inference_steps, seed,
                  progress=gr.Progress()):
    progress(0, desc="Loading weights...")
    lcm_pipe = get_stitch_sdxl_lcm_pipeline(small_sd, large_sd)
    generator = torch.manual_seed(seed)
    latents = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)
    progress(0, desc="Generating images...")
    image = lcm_pipe(prompt, negative_prompts=negative_prompts, unet_s_ratio=r, num_inference_steps=num_inference_steps,
                     latents=latents, height=512, width=512, guidance_scale=guidance_scale, generator=generator).images[0]
    yield image


ratios = [round(item, 2) for item in np.arange(0, 1.1, 0.1)]

with gr.Blocks() as sd_14_15_demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                inputs = gr.Textbox(lines=2, placeholder="prompt here...")
                stitch_ratio = gr.Dropdown(
                    choices=ratios, value=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], multiselect=True, label="T-Stitch Ratio",
                    info="The fraction of small SD model"
                )
                small_sd = gr.Dropdown(
                    ["nota-ai/bk-sdm-tiny", "nota-ai/bk-sdm-small"], value=["nota-ai/bk-sdm-tiny"], label="Small Model",
                    info="The smaller SD model"
                )
                large_sd = gr.Dropdown(
                    ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "nota-ai/bk-sdm-base",
                     "Envvi/Inkpunk-Diffusion", "nitrosocke/Ghibli-Diffusion"], value=["nitrosocke/Ghibli-Diffusion"],
                    label="Large Model", info="The larger SD model"
                )
                guidance_scale = gr.Slider(1, 50, label='Guidance scale')  # guidance_scale
                num_inference_steps = gr.Slider(4, 50, step=1, label="Sampling Steps")  # num_inference_steps
                seed = gr.Slider(0, 10000, step=1, label="Seed")  # seed

                with gr.Row():
                    clear_btn = gr.ClearButton(
                        components=[inputs, stitch_ratio, small_sd, large_sd, guidance_scale, num_inference_steps,
                                    seed])
                    generate_btn = gr.Button(value="Generate")

            with gr.Column():
                outputs = gr.Gallery(show_label=True, label="Generated Images")

            generate_btn.click(
                fn=sd_14_15_func,
                inputs=[
                    inputs,
                    stitch_ratio,
                    small_sd,
                    large_sd,
                    guidance_scale,
                    num_inference_steps,
                    seed
                ],
                outputs=outputs,
            )

        with gr.Row():
            examples = gr.Examples(
                examples=[
                    ["A ghibli style princess with golden hair in New York City", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     "nota-ai/bk-sdm-tiny", "nitrosocke/Ghibli-Diffusion", 7.0, 50, 4],
                    ["ghibli style beautiful Caribbean beach tropical (sunset)", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     "nota-ai/bk-sdm-tiny", "nitrosocke/Ghibli-Diffusion", 7.0, 50, 666],
                    ["a squirrel in the park, nvinkpunk style", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "nota-ai/bk-sdm-tiny",
                     "Envvi/Inkpunk-Diffusion", 7.0, 50, 1024],
                    ["A polar bear on mars, nvinkpunk style", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "nota-ai/bk-sdm-tiny",
                     "Envvi/Inkpunk-Diffusion", 7.0, 50, 2025],
                    [
                        "Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage.",
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "nota-ai/bk-sdm-tiny",
                        "CompVis/stable-diffusion-v1-4", 7.0, 50, 6666],
                ],
                inputs=[inputs, stitch_ratio, small_sd, large_sd, guidance_scale, num_inference_steps, seed],
            )

with gr.Blocks() as sdxl_demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                inputs = gr.Textbox(lines=2, placeholder="prompt here...")
                negative_prompts = gr.Textbox(lines=2, placeholder="negative prompt here...")
                stitch_ratio = gr.Dropdown(
                    choices=ratios, value=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], multiselect=True, label="T-Stitch Ratio",
                    info="The fraction of small SD model"
                )
                small_sd = gr.Dropdown(
                    ["segmind/SSD-1B"], value=["segmind/SSD-1B"], label="Small Model", info="The smaller SD model",
                )
                large_sd = gr.Dropdown(
                    ["stabilityai/stable-diffusion-xl-base-1.0"], value=["stabilityai/stable-diffusion-xl-base-1.0"],
                    label="Large Model", info="The larger SD model"

                )
                guidance_scale = gr.Slider(1, 50, label='Guidance scale')  # guidance_scale
                num_inference_steps = gr.Slider(4, 50, step=1, label="Sampling Steps")  # num_inference_steps
                seed = gr.Slider(0, 10000, step=1, label="Seed")  # seed

                with gr.Row():
                    clear_btn = gr.ClearButton(
                        components=[inputs, negative_prompts, stitch_ratio, small_sd, large_sd, guidance_scale,
                                    num_inference_steps,
                                    seed])
                    generate_btn = gr.Button(value="Generate")

            with gr.Column():
                outputs = gr.Gallery(show_label=True, label="Generated Images")

            generate_btn.click(
                fn=sdxl_func,
                inputs=[
                    inputs,
                    negative_prompts,
                    stitch_ratio,
                    small_sd,
                    large_sd,
                    guidance_scale,
                    num_inference_steps,
                    seed
                ],
                outputs=outputs,
            )

        with gr.Row():
            examples = gr.Examples(
                examples=[
                    [
                        "concept art of dragon flying over town, clouds. digital artwork, illustrative, painterly, matte painting, highly detailed, cinematic composition",
                        "photo, photorealistic, realism, ugly", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        "segmind/SSD-1B", "stabilityai/stable-diffusion-xl-base-1.0", 7.0, 50, 4],
                ],
                inputs=[inputs, negative_prompts, stitch_ratio, small_sd, large_sd, guidance_scale, num_inference_steps,
                        seed],
            )

with gr.Blocks() as sdxl_lcm_demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                inputs = gr.Textbox(lines=2, placeholder="prompt here...", every=1)
                negative_prompts = gr.Textbox(lines=2, placeholder="negative prompt here...")
                stitch_ratio = gr.Slider(0, 1, step=0.1, label="T-Stitch Ratio")  # stitch_ratio
                small_sd = gr.Dropdown(
                    ["segmind/SSD-1B"], value="segmind/SSD-1B", label="Small Model",
                    info="The smaller SD model"
                )
                large_sd = gr.Dropdown(
                    ["stabilityai/stable-diffusion-xl-base-1.0"], value="stabilityai/stable-diffusion-xl-base-1.0",
                    label="Large Model", info="The larger SD model"
                )
                guidance_scale = gr.Slider(0, 20, label='Guidance scale')  # guidance_scale
                num_inference_steps = gr.Slider(1, 8, step=1, label="Sampling Steps")  # num_inference_steps
                seed = gr.Slider(0, 10000, step=1, label="Seed")  # seed

                with gr.Row():
                    clear_btn = gr.ClearButton(
                        components=[inputs, negative_prompts, stitch_ratio, small_sd, large_sd, guidance_scale,
                                    num_inference_steps,
                                    seed])
                    generate_btn = gr.Button(value="Generate")

            with gr.Column():
                outputs_img = gr.Image(label="Generated Image")

            generate_btn.click(
                fn=sdxl_lcm_func,
                inputs=[
                    inputs,
                    negative_prompts,
                    stitch_ratio,
                    small_sd,
                    large_sd,
                    guidance_scale,
                    num_inference_steps,
                    seed
                ],
                outputs=outputs_img,
            )

with gr.Blocks() as main_page:
    with gr.Column():
        gr.HTML("""
            <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">T-Stitch: Accelerating Sampling in Pre-trained Diffusion Models with Trajectory Stitching</h1>
            <div align="center">
            <img align="center" src='file/.github/image-20231011133541606.png' width="70%">
            <h3> T-Stitch first leverages a smaller DPM in the initial steps as a cheap drop-in replacement of the larger DPM and switches to the larger DPM at a later stage, thus achieving flexible speed and quality trade-offs.</h3>
            </div>
            """)
        tabbed_page = gr.TabbedInterface([sd_14_15_demo, sdxl_demo, sdxl_lcm_demo],
                                         ["Stable Diffusion 1.4/1.5/Stylized", "SDXL", "SDXL-LCM"])

main_page.launch(allowed_paths=['./'])
