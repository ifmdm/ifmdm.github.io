"""
Usage:

    CUDA_VISIBLE_DEVICES=0,1 python train_stage1.py \
        --dataset celebvhq \
        --exp_path ./output \
        --exp_name LIA_033_refine_block \
        --batch_size 16 \
        --num_workers 16 \
        --lr 0.0001  \
        --latent_dim_motion 20 \
        --resume_ckpt ./checkpoints/stage1_022_best_1529000.pt \
        --refine_block \
        --resume_not_strict \
        2>&1 | tee output/train.log
"""
import argparse
import os
import shutil
import torch
import io
import webdataset as wds
from torch.utils import data
from dataset import Vox256, Taichi, TED, CelebVHQ
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from time import time
import logging
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def visualize_mask(mask):
    assert mask.shape[1] == 4
    new_mask = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3]).to(mask.device)
    new_mask[:, 0, :, :] = torch.where(mask[:, 0, :, :] == 1, torch.tensor(0.5), torch.tensor(0.0))
    new_mask[:, 1, :, :] = torch.where(mask[:, 0, :, :] == 1, torch.tensor(0.5), torch.tensor(0.0))
    new_mask[:, 2, :, :] = torch.where(mask[:, 0, :, :] == 1, torch.tensor(0.5), torch.tensor(0.0))
    
    new_mask[:, 0, :, :] = torch.where(mask[:, 1, :, :] == 1, torch.tensor(1.0), new_mask[:, 0, :, :])
    new_mask[:, 1, :, :] = torch.where(mask[:, 2, :, :] == 1, torch.tensor(1.0), new_mask[:, 1, :, :])
    new_mask[:, 2, :, :] = torch.where(mask[:, 3, :, :] == 1, torch.tensor(1.0), new_mask[:, 2, :, :])
    return new_mask


def save_valid_image(imgs, prefix):
    for i in range(len(imgs)):
        assert len(imgs[0]) == len(imgs[i])
    torchvision.utils.save_image(
        torch.cat(imgs, dim=0),
        f"{prefix}.png",
        nrow=len(imgs[0]),
        padding=0,
        normalize=True,
        value_range=(-1, 1),
    )


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


def write_loss(i, vgg_loss, l1_loss, g_loss, d_loss, writer):
    writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    writer.add_scalar('l1_loss', l1_loss.item(), i)
    writer.add_scalar('gen_loss', g_loss.item(), i)
    writer.add_scalar('dis_loss', d_loss.item(), i)
    writer.flush()


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


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def process_sample(sample):
    source_image = sample['source.pth']
    target_image = sample['target.pth']
    source_image = io.BytesIO(source_image)
    target_image = io.BytesIO(target_image)
    source_image = torch.load(source_image, weights_only=True)
    target_image = torch.load(target_image, weights_only=True)
    return source_image, target_image


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # setup logger
    if rank == 0:
        logger = create_logger(f"{args.exp_path}/{args.exp_name}")
    else:
        logger = create_logger(None)

    # setsup tensorboard
    writer = SummaryWriter(log_path)

    logger.info('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    if args.dataset == 'ted':
        dataset = TED('train', transform, True)
        dataset_test = TED('test', transform)
    elif args.dataset == 'vox':
        dataset = Vox256('train', transform, False)
        dataset_test = Vox256('test', transform)
    elif args.dataset == 'taichi':
        dataset = Taichi('train', transform, True)
        dataset_test = Taichi('test', transform)
    elif args.dataset == 'celebvhq':
        if args.shard_path is not None:
            dataset = wds.WebDataset(
                args.shard_path,
                resampled=True,
            ).map(process_sample)
        else:
            dataset = CelebVHQ('train', transform, True, csv_path=[
                "./datasets/CelebV-Text-and-HDTF/stage1_20240820_celebvtext.csv",
                "./datasets/CelebV-Text-and-HDTF/stage1_20240820_hdtf.csv",
            ], length_multiplier=1)
        dataset_test = CelebVHQ('test', transform, csv_path=[
            "./datasets/CelebV-Text-and-HDTF/stage1_20240820_celebvtext.csv",
            "./datasets/CelebV-Text-and-HDTF/stage1_20240820_hdtf.csv",
        ])
    else:
        raise NotImplementedError

    if args.shard_path is not None:
        loader = wds.WebLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size // world_size,
            shuffle=False,
            pin_memory=True,
        )
    else:
        loader = data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size // world_size,
            sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=True,
        )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=args.num_workers,
        batch_size=8,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    logger.info('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        logger.info('==> resume from iteration %d' % (args.start_iter))
    
    
    logger.info('==> training')
    best_l1_loss_value = 1000
    start_time = time()
    log_steps = 0
    pbar = range(args.iter)
    for idx in pbar:
        i = idx + args.start_iter
        log_steps += 1

        # laoding data
        img_source, img_target = next(loader)
        img_source = img_source.to(rank, non_blocking=True)
        img_target = img_target.to(rank, non_blocking=True)

        img_source = trainer.blur_background(img_source)
        img_target = trainer.blur_background(img_target)

        # update generator
        vgg_loss, l1_loss, gan_g_loss, img_recon = trainer.gen_update(img_source, img_target)

        # update discriminator
        gan_d_loss = trainer.dis_update(img_target, img_recon)

        if rank == 0:
            # write to log
            write_loss(idx, vgg_loss, l1_loss, gan_g_loss, gan_d_loss, writer)
            writer.flush()

        # display
        if (i == args.start_iter or i % args.display_freq == 0) and rank == 0:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            log_steps = 0

            logger.info("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f] [steps/sec: %f]"
                        % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item(), steps_per_sec))
            
            torch.cuda.synchronize()
            start_time = time()

            save_valid_image([
                img_source,
                img_target,
                img_recon,
            ], prefix=f"{log_path}/train_step{i:06d}")

        # save model
        if (i == args.start_iter or i % args.save_freq == 0) and rank == 0:
            val_l1_losses = []
            for j, (img_test_source, img_test_target) in enumerate(loader_test):
                if j >= 8: break
                img_test_source = img_test_source.to(rank, non_blocking=True)
                img_test_target = img_test_target.to(rank, non_blocking=True)

                img_test_source = trainer.blur_background(img_test_source)
                img_test_target = trainer.blur_background(img_test_target)

                img_test_source = torch.cat([img_test_source, img_test_source], dim=0)
                img_test_target = torch.cat([img_test_target, img_test_target[1:], img_test_target[0:1]], dim=0)

                img_recon, img_source_ref = trainer.sample(img_test_source, img_test_target)

                val_l1_loss = torch.nn.L1Loss()(img_recon, img_test_target)
                val_l1_losses.append(val_l1_loss.item())

                save_valid_image([
                    img_test_source,
                    img_test_target,
                    img_recon,
                    img_source_ref,
                ], prefix=f"{log_path}/valid_step{i:06d}_index{j:06d}")

                torchvision.utils.save_image((img_test_source[0] + 1) / 2, f"{log_path}/valid_step{i:06d}_index{j:06d}_source.png")
                torchvision.utils.save_image((img_test_target[0] + 1) / 2, f"{log_path}/valid_step{i:06d}_index{j:06d}_target.png")
                torchvision.utils.save_image((img_recon[0] + 1) / 2, f"{log_path}/valid_step{i:06d}_index{j:06d}_recon.png")
                torchvision.utils.save_image((img_source_ref[0] + 1) / 2, f"{log_path}/valid_step{i:06d}_index{j:06d}_source_ref.png")
            avg_val_l1_loss = sum(val_l1_losses) / len(val_l1_losses)
            if avg_val_l1_loss < best_l1_loss_value:
                best_l1_loss_value = avg_val_l1_loss
                trainer.save(i, checkpoint_path, is_best=True)
                logger.info(f"The best val l1 loss is updated: {best_l1_loss_value:.6f}")
            else:
                logger.info(f"The current val l1 loss is: {avg_val_l1_loss:.6f}")
                
            trainer.save(i, checkpoint_path)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1500000)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_with_custom_rule", action='store_true')
    parser.add_argument("--resume_not_strict", action='store_true')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./exps/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--decoder_only", action='store_true')
    parser.add_argument("--lambda_gan_g_loss", type=float, default=1.0)
    parser.add_argument("--super_resolution_training", action='store_true')
    parser.add_argument("--refine_block", action='store_true')
    parser.add_argument("--shard_path", type=str, default=None)
    opts = parser.parse_args()

    exp_dir = os.path.join(opts.exp_path, opts.exp_name)
    if os.path.exists(exp_dir):
        if input(f"Experiment {exp_dir} already exists. Remove[y/n]? : ") == 'y':
            shutil.rmtree(exp_dir)

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        world_size = n_gpus
        print('==> training on %d gpus' % n_gpus)
        mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
    elif n_gpus == 1:
        print('==> training on single gpu')
        main(0, 1, opts)
    else:
        raise NotImplementedError
