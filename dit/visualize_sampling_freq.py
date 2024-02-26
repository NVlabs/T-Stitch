# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import math
from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def get_fourier_spectrum(latent):
    latent = latent.cpu()
    if len(latent.shape) == 3:  # for ViT
        b, n, c = latent.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
    elif len(latent.shape) == 4:  # for CNN
        b, c, h, w = latent.shape
    else:
        raise Exception("shape: %s" % str(latent.shape))
    latent = fourier(latent)
    latent = shift(latent).mean(dim=(0, 1))
    latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
    return latent

def get_fourier_spectrum_mine(latent):
    latent = latent[None, ...]
    b, c, h, w = latent.shape
    latent = fourier(latent)
    latent = shift(latent).mean(dim=(0, 1))
    latent = latent.diag()[int(h / 2):]
    return latent


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # randomly sample some class labels, 32 in this case
    class_labels = torch.randint(0, 1000, (32, )).tolist()

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.ddim_sample_loop_progressive(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
        device=device
    )
    latents = [sample['sample'].chunk(2, dim=0)[0] for sample in samples]

    latents = torch.stack(latents, dim=0)
    model_tag = args.model.split("/")[0]
    torch.save(latents, f'{model_tag}_step_images_steps_{args.num_sampling_steps}.pt')

    # visualize log amplitudes
    diff_steps = list(reversed(diffusion.timestep_map))
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
    traj_len, batch_size, C, H, W = latents.shape
    for step in range(traj_len):
        batch = latents[step]

        # adopt from https://github.com/xxxnell/how-do-vits-work/blob/transformer/fourier_analysis.ipynb, thanks!
        latent = get_fourier_spectrum(batch)

        freq = np.linspace(0, 1, len(latent))
        ax1.plot(freq, latent, color=cm.plasma_r(step / traj_len), label=f't = {diff_steps[step]}')

    ax1.set_xlim(left=0, right=1)
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Log amplitude')

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))
    ax1.grid(linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--solver", type=str, default="ddpm")
    parser.add_argument("--ckpt", type=str, default='pretrained_models/dit_s_256.pt')
    args = parser.parse_args()
    main(args)