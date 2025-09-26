import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from vgg19 import VGGLoss
from networks.stage1_model import Discriminator
from networks.stage1_model import Generator
from transformers import AutoModelForImageSegmentation


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).to(
            device)
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        if args.super_resolution_training:
            g_params = []
            for n, p in self.gen.named_parameters():
                if \
                    'enc.net_app.convs.0' in n or \
                    'enc.net_app.convs.1' in n or \
                    'dec.convs.12' in n or \
                    'dec.convs.13' in n or \
                    'dec.to_rgbs.6' in n or \
                    'dec.to_flows.6' in n:
                    print(f"[DEBUG] [included] n: {n}, p.shape: {p.shape}")
                    g_params.append(p)
                else:
                    print(f"[DEBUG] [not included] n: {n}, p.shape: {p.shape}")
            self.g_optim = optim.Adam(
                g_params,
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )
        elif args.refine_block:
            g_params = []
            for n, p in self.gen.named_parameters():
                if 'refine_block' in n:
                    print(f"[DEBUG] [included] n: {n}, p.shape: {p.shape}")
                    g_params.append(p)
                else:
                    print(f"[DEBUG] [not included] n: {n}, p.shape: {p.shape}")
            self.g_optim = optim.Adam(
                g_params,
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )
        elif args.decoder_only:
            g_params = []
            for n, p in self.gen.named_parameters():
                if 'dec' in n:
                    g_params.append(p)
            self.g_optim = optim.Adam(
                g_params,
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )
        else:
            self.g_optim = optim.Adam(
                self.gen.parameters(),
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
            )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(rank)
        self.lambda_gan_g_loss = args.lambda_gan_g_loss

        self.decoder_only = args.decoder_only

        self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet-portrait', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.birefnet.cuda()
        self.birefnet.eval()
    
    def blur_background(self, image, kernel_size=21):
        _, _, H, W = image.shape
        # Load image
        image_size = (1024, 1024)
        transform_image = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_image = transform_image(image)
        # Predict
        with torch.no_grad():
            mask = self.birefnet(input_image)[-1].sigmoid()
            mask = torch.nn.functional.interpolate(mask, size=(H, W))
        # Gaussian blur for the image with the mask
        blur_image = torchvision.transforms.functional.gaussian_blur(image, kernel_size=kernel_size)
        output_image = torch.where(mask > 0.5, image, blur_image)
        return output_image

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def get_mask(self, img_target):
        with torch.no_grad():
            img_for_detector = img_target
            img_for_detector = img_for_detector * 0.5 + 0.5
            img_for_detector = torch.nn.functional.interpolate(img_for_detector, (256, 256), mode='bilinear')
            img_target_mask = self.detector_awing.get_masks(img_for_detector)
            img_target_mask = torch.nn.functional.interpolate(img_target_mask, (img_target.shape[2], img_target.shape[3]), mode='nearest')
        return img_target_mask

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        img_target_recon = self.gen(img_source, img_target, None)
        img_recon_pred = self.dis(img_target_recon)

        l1_loss = F.l1_loss(img_target_recon, img_target)
        vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)

        g_loss = vgg_loss + l1_loss + gan_g_loss * self.lambda_gan_g_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, img_target_recon

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)
    
        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()

            img_recon = self.gen(img_source, img_target, None)
            img_source_ref = self.gen(img_source, None, None)

        return img_recon, img_source_ref
    
    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        ckpt_name = os.path.basename(resume_ckpt)
        if "best" in ckpt_name:
            ckpt_name = ckpt_name.split("_")[-1]
            start_iter = int(os.path.splitext(ckpt_name)[0])
        elif os.path.splitext(ckpt_name)[0].isdigit():
            start_iter = int(os.path.splitext(ckpt_name)[0])
        else:
            start_iter = 0

        if self.args.resume_with_custom_rule:
            print("[INFO] resume with custom rule")
            self.gen.module.load_state_dict(ckpt["gen"])
        elif self.args.resume_not_strict:
            print("[INFO] resume with not strict rule")
            self.gen.module.load_state_dict(ckpt["gen"], strict=False)
            self.dis.module.load_state_dict(ckpt["dis"], strict=False)
        else:
            print("[INFO] resume with default rule")
            self.gen.module.load_state_dict(ckpt["gen"])
            self.dis.module.load_state_dict(ckpt["dis"])
            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path, is_best=False):
        if is_best:
            checkpoint_path = f"{checkpoint_path}/best_{str(idx).zfill(6)}.pt"
        else:
            checkpoint_path = f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            checkpoint_path,
        )