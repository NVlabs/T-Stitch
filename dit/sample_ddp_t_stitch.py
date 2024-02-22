# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def three_combo(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    dit_names = ['DiT-S/2', 'DiT-B/2', 'DiT-XL/2']
    ckpt_paths = [args.s_ckpt, args.b_ckpt, args.xl_ckpt]
    denoisers = []
    for i, name in enumerate(dit_names):
        dit_m = DiT_models[name](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        dit_m.cfg_scale = args.cfg_scale
        state_dict = find_model(ckpt_paths[i])
        dit_m.load_state_dict(state_dict)
        denoisers.append(dit_m)

    from models import MultiTStitch
    model = MultiTStitch(denoisers).to(device)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    small_candidates = [round(item, 1) for item in np.arange(0., 1.1, 0.1).tolist()]

    for small_ratio in small_candidates:

        base_candidates = [round(item, 1) for item in np.arange(0., (1 - small_ratio) + 0.1, 0.1).tolist()]

        for base_ratio in base_candidates:
            if small_ratio + base_ratio > 1.0:
                break
            xl_ratio = round(1 - (small_ratio + base_ratio), 1)
            model.alloc = [small_ratio, base_ratio, xl_ratio]

            print(f"small_ratio: {small_ratio}, base_ratio: {base_ratio}, xl_ratio: {xl_ratio}")

            # Create folder to save samples:
            # ckpt_string_name = os.path.basename(args.b_ckpt).replace(".pt", "") if args.s_ckpt else "pretrained"
            folder_name = f"seed-{args.global_seed}-images-{args.num_fid_samples}-solver-{args.solver}-steps-{args.num_sampling_steps}-small-{small_ratio}-base-{base_ratio}-xl-{xl_ratio}"
            sample_folder_dir = f"{args.sample_dir}/{folder_name}"

            if rank == 0:
                os.makedirs(sample_folder_dir, exist_ok=True)
                print(f"Saving .png samples at {sample_folder_dir}")
            dist.barrier()

            # continue

            # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
            n = args.per_proc_batch_size
            global_batch_size = n * dist.get_world_size()
            # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
            total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
            if rank == 0:
                print(f"Total number of images that will be sampled: {total_samples}")
            assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
            samples_needed_this_gpu = int(total_samples // dist.get_world_size())
            assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
            iterations = int(samples_needed_this_gpu // n)
            pbar = range(iterations)
            pbar = tqdm(pbar) if rank == 0 else pbar
            total = 0
            for _ in pbar:
                # Sample inputs:
                z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
                y = torch.randint(0, args.num_classes, (n,), device=device)

                # Setup classifier-free guidance:
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                    sample_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    sample_fn = model.forward

                # Sample images:
                if args.solver == 'ddpm':
                    samples = diffusion.p_sample_loop(
                        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                        device=device
                    )
                elif args.solver == 'ddim':
                    samples = diffusion.ddim_sample_loop(
                        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                        device=device
                    )
                else:
                    pass

                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

                samples = vae.decode(samples / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                              dtype=torch.uint8).numpy()

                # Save samples to disk as individual .png files
                for i, sample in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
                total += global_batch_size

            # Make sure all processes have finished saving their samples before attempting to convert to .npz
            dist.barrier()
            if rank == 0:
                create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
                print("Done.")
            dist.barrier()

    dist.destroy_process_group()


def two_combo(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    dit_names = ['DiT-S/2', 'DiT-XL/2']
    ckpt_paths = [args.s_ckpt, args.xl_ckpt]
    denoisers = []
    for i, name in enumerate(dit_names):
        dit_m = DiT_models[name](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        dit_m.cfg_scale = args.cfg_scale
        state_dict = find_model(ckpt_paths[i])
        dit_m.load_state_dict(state_dict)
        denoisers.append(dit_m)

    from models import TStitch
    model = TStitch(denoisers, ratio=args.ratio).to(device)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0


    # enumerate all possible fractions for the small denoiser with a step size of 0.1
    candidate_ratios = [round(item, 1) for item in np.arange(0., 1.1, 0.1).tolist()]

    for small_ratio in candidate_ratios:
        model.ratio = small_ratio

        print(f"small_ratio: {small_ratio}")

        # Create folder to save samples:
        ckpt_string_name = os.path.basename(args.b_ckpt).replace(".pt", "") if args.s_ckpt else "pretrained"
        folder_name = f"{ckpt_string_name}-{args.image_size}-vae-{args.vae}-" \
                      f"cfg-{args.cfg_scale}-seed-{args.global_seed}-images-{args.num_fid_samples}-solver-{args.solver}-ratio-{small_ratio}"
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
        if rank == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
            print(f"Saving .png samples at {sample_folder_dir}")
        dist.barrier()

        # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
        n = args.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
        total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
        pbar = range(iterations)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0
        for _ in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            if args.solver == 'ddpm':
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
            elif args.solver == 'ddim':
                samples = diffusion.ddim_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
            else:
                pass

            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                          dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size

        # Make sure all processes have finished saving their samples before attempting to convert to .npz
        dist.barrier()
        if rank == 0:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--solver", type=str, default="ddim")
    parser.add_argument("--s_ckpt", type=str, default='pretrained_models/dit_s_256.pt')
    parser.add_argument("--b_ckpt", type=str, default='pretrained_models/dit_b_256.pt')
    parser.add_argument("--xl_ckpt", type=str, default='pretrained_models/dit_xl_256.pt')
    parser.add_argument("--three_combo", action='store_true')
    args = parser.parse_args()
    if args.three_combo:
        three_combo(args)
    else:
        two_combo(args)
