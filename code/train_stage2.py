"""
A minimal training script for DiT using PyTorch DDP.

Usage:

    CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=23455 train_stage2.py \
        --exp-prefix v241110 \
        --results-dir ./output \
        --model "DiT-A0" \
        --epochs 100000 \
        --lr 0.0001 \
        --num-workers 24 \
        --global-batch-size 128 \
        --global-seed 903 \
        --ckpt-every 50000 \
        --diffusion_steps 1000 \
        --latent_dim_motion 20 \
        2>&1 | tee output/train_motion_gen.log
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from networks.dit_models_v241024 import DiT_models
from diffusion import create_diffusion
from dataset import CelebVHQ_motion
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        print(f"[DEBUG] Creating experiment directory at {args.results_dir}")
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{args.exp_prefix}-{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        print(f"[DEBUG] Creating experiment directory at {experiment_dir}")
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = DiT_models[args.model](
        in_channels=args.latent_dim_motion,
    )
    # resume ckpt
    if args.resume_ckpt is not None:
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        new_checkpoint = dict()
        model_checkpoint = model.named_parameters()
        for name, param in model_checkpoint:
            if name in checkpoint['model']:
                if checkpoint['model'][name].shape == param.shape:
                    print(f"[included] copy {name} from checkpoint: p.shape: {param.shape}, cp.shape: {checkpoint['model'][name].shape}")
                    new_checkpoint[name] = checkpoint['model'][name]
                else:
                    print(f"[not included] skip {name} due to shape mismatch: p.shape: {param.shape}, cp.shape: {checkpoint['model'][name].shape}")
                    continue
            else:
                print(f"[DEBUG] {name} not in checkpoint")
                continue
        model.load_state_dict(new_checkpoint, strict=False)
        # model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"Loaded checkpoint from {args.resume_ckpt}")
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(
        timestep_respacing="",
        diffusion_steps=args.diffusion_steps,
    )  # default: linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )
    dataset = CelebVHQ_motion(
        split='train',
        csv_path=[
            './datasets/CelebV-Text-and-HDTF/stage2_20240917_hdtf.csv',
            './datasets/CelebV-Text-and-HDTF/stage2_20240917_celebvtext.csv',
        ],
        cond_mask=[
            1,
            1,
        ],
        transform=transform,
        length_multiplier=10,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    batch_size = int(args.global_batch_size // dist.get_world_size())
    print(f"[INFO] batch_size: {batch_size}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # # test data
    # if rank == 0:
    #     tbar = tqdm(loader, total=len(loader), desc='test dataloader')
    #     error_ratio = 0
    #     for i, data in enumerate(tbar):
    #         _, _, flags = data
    #         error_ratio = error_ratio * i / (i + 1) + (flags.sum() / flags.numel()).item() / (i + 1)
    #         tbar.set_postfix(error_ratio=error_ratio)
    #     print(f"[DEBUG] error ratio: {error_ratio}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if args.resume_ckpt is not None:
        start_train_steps = args.resume_ckpt.split('/')[-1]
        start_train_steps = start_train_steps.split('.')[0]
        if start_train_steps.isdigit():
            start_train_steps = int(start_train_steps)
        elif start_train_steps.split('_')[-1].isdigit():
            start_train_steps = int(start_train_steps.split('_')[-1])
        else:
            raise ValueError(f"Invalid checkpoint path: {args.resume_ckpt}")
        train_steps += start_train_steps
    else:
        start_train_steps = 0

    best_train_loss = float('inf')
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for i, (meta, data) in enumerate(loader):
            # Get data
            x = data['x'].to(device)
            y = data['aud_target'].to(device)
            # Training step
            x_hint = x[:,0:1,:]
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True)
            x = x[:,1:,:]
            y = y[:,2:,:]
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, x_hint=x_hint, x_mean=x_mean, x_std=x_std)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            if train_steps % args.log_every == 0 or train_steps == start_train_steps:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    if rank == 0:
                        logger.info(f"New best train loss: {best_train_loss:.4f}")
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 or train_steps == start_train_steps:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            train_steps += 1

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--exp-prefix", type=str, default="s16")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--detector_awing_model_path", type=str, default='./checkpoints/WFLW_4HG.pth')
    parser.add_argument("--gen_model_path", type=str, default='./output/LIA_009_celebvtext/checkpoint/1990000.pt')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    args = parser.parse_args()
    main(args)
