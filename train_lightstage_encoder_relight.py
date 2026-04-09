from random import random, randint
import os
import numpy as np
import torch
import click
import json
import glob
import copy
import torch.distributed as dist
import torchvision
import pickle
import math
import re
from distutils.util import strtobool
from natsort import natsorted

import argparse
import cv2
from tqdm import tqdm
import wandb

from torch import nn, autograd, optim
from torch.utils import data
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch_utils import misc
import dnnlib
from dnnlib.seg_tools import *
import legacy
from training.networks import Net
from configs.swin_config import get_config
from training.volumetric_rendering.rendering_utils import sample_camera_positions, create_cam2world_matrix
# from training.volumetric_rendering.rendering_utils import LookAtPoseSampler, FOV_to_intrinsics
# from training.volumetric_rendering.rendering_utils import GaussianCameraPoseSampler, RealGaussianCameraPoseSampler
# from inversion.criteria import id_loss as IDLoss
from inversion.criteria import mrf_loss as IDMRFLoss

from training.triplane import TriPlaneGenerator
# from training.dual_discriminator import DualDiscriminator
from inversion.utils.debugger import set_trace
# from image_utils.tonemap import apply_tonemap as process_image

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
tonemap_olat = lambda x: pow(x, 0.5)
# tonemap_olat = lambda x: np.clip(pow(np.clip(x, 0, 5), 0.5) * 1.5, 0, 1)


def torch2numpy(img, drange, nimgs=4, use_tonemap=False):
    img = img[0:nimgs].data.cpu().clamp(-1, 65535)
    img = F.interpolate(img, (512, 512)).permute(0, 2, 3, 1)
    _, H, W, C = img.shape
    img = ((img * drange / 2.) + drange / 2.).numpy()[:, :, :, ::-1]
    if use_tonemap:
        img = tonemap_olat(img)
    return np.hstack(img)


def save_image_grid(img: list, fname: str):
    img_grid = np.vstack(img)
    cv2.imwrite(fname, img_grid)


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def launch_training(desc, outdir):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def cam_sampler(batch, device):
    """
    Horizontal Mean : Mean yaw angle in degrees
    Vertical Mean: Mean pitch angle in degrees
    """
    camera_points, phi, theta = sample_camera_positions(device, n=batch, r=2.7, horizontal_mean=0.5*math.pi,
                                                        vertical_mean=0.5*math.pi, horizontal_stddev=0.3,
                                                        vertical_stddev=0.155, mode='gaussian')
    c = create_cam2world_matrix(-camera_points, camera_points, device=device)
    c = c.reshape(batch, -1)
    c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(batch, 1).to(c)), -1)
    return c


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

    
def requires_grad(model, flag=True, debug=False):
    if debug:
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

    for _, p in model.named_parameters():
        p.requires_grad = flag

    if debug:
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

        set_trace()
        print("Done!")

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


class HDRLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_loss  = nn.L1Loss()

    def forward(self, source, target, clamp=False):
        # source = source.clamp(0, 1.0) if clamp is True else source
        l2_val = (source - target) ** 2
        scaling_grad = (1. / (1e-3 + source)).detach()
        return (l2_val * (scaling_grad**2)).mean()


class DepthLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target, weight=1):
        high, low = source.min(), source.max()
        source_norm = (source-low)/((high-low)+1e-8)
        target_norm = (target-low)/((high-low)+1e-8)

        loss = self.criterion(source_norm, target_norm)
        return loss * weight


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5, wt_mode=0):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        if wt_mode == 0:
            self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        elif wt_mode == 1:
            self.weights = (1.0, 0.1, 0.0001, 0.0, 0.0)
        elif wt_mode == 2:
            self.weights = (1.0, 1.0, 0.1, 0.10, 0.0001)
        else:
            raise NotImplementedError

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        # source, target = (source + 1) / 2, (target + 1) / 2
        source = (source-self.mean) / self.std
        target = (target-self.mean) / self.std
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)

        return loss 


def linear2rgb(img, drange):
    tonemap_disp = lambda x: (pow(x / (pow(2, 16)), 0.5) * 255)
    img = img.clamp(-1, 65535)
    img = ((img * drange / 2.) + drange / 2.)
    img = tonemap_disp(img)
    return img


def train(rank, world_size, opts):
    outdir             = opts.outdir
    g_ckpt             = opts.G_ckpt
    max_steps          = opts.max_steps
    batch              = opts.batch
    wt_col             = opts.wt_col
    wt_lpips           = opts.wt_lpips
    train_real         = opts.train_real
    gen_interval       = opts.gen_interval
    tensorboard        = opts.tensorboard
    print_interval     = opts.print_interval
    start_stage2_after = opts.start_stage2_after
    update_sr_after    = opts.update_sr_after
    port               = opts.port
    reload_modules     = opts.reload
    num_gpus           = opts.num_gpus
    random_seed        = opts.random_seed
    local_rank         = opts.local_rank
    relight            = opts.relight
    use_tonemap        = opts.tonemap
    use_hdr_loss       = opts.use_hdr_loss
    train_gen          = opts.train_gen
    gen_nrr            = opts.gen_nrr
    lpips_wt_mode      = opts.lpips_wt_mode
    num_ids            = opts.num_ids
    debug_mode         = opts.debug_mode
    dataset_name       = opts.dataset_name
    lightstage_res     = opts.lightstage_res
    num_views          = opts.num_views
    #
    # train_disc         = opts.train_disc
    # d_lr               = opts.d_lr
    # use_discriminator  = opts.use_disc
    # crop_size          = opts.crop_size
    # reg_interval       = opts.reg_interval
    # r1_gamma           = opts.r1_gamma
    # lr                 = opts.lr
    # wt_adv_mv          = opts.wt_adv_mv
    # img_ch             = opts.img_ch
    # e_ckpt             = opts.E_ckpt
    # afa_ckpt           = opts.AFA_ckpt
    # r_ckpt             = opts.R_ckpt
    # enc_nrr            = opts.enc_nrr
    # arch               = opts.arch
    # act_fn             = opts.act_fn
    # wt_real            = opts.wt_real
    # wt_id              = opts.wt_id
    # wt_adv_ref         = opts.wt_adv_ref
    # wt_tri             = opts.wt_tri
    # ref_interval       = opts.ref_interval
    
    if rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="goae-inversion-olat",
            name=f'{opts.outdir}-{opts.port}',

            # track hyperparameters and run metadata
            config={
                "learning_rate": opts.lr,
                "gpus": opts.num_gpus,
                "batch_size": opts.batch,
                "num_ids": opts.num_ids,
                "o_dim": opts.o_dim,
                "lpips_wt_mode": opts.lpips_wt_mode,
                "lpips_wt": opts.wt_lpips,
                "lcol_wt": opts.wt_col,
                "expt_id": outdir,
                "decoder_depth": opts.decoder_depth,
                "HDR_Loss":opts.use_hdr_loss,
                "viewdirs":opts.use_viewdirs,
            }
        )
    
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)

    setup(rank, world_size, port)
    device = torch.device(rank)

    # conv2d_gradfix.enabled = True  # Improves training speed.
    
    # load the pre-trained model
    print('Loading generator from "%s"...' % g_ckpt)
    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = gen_nrr
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    if use_tonemap:
        OLAT_SCALE = 255.
        apply_tonemap  = False
    else:
        OLAT_SCALE = 65535.
        apply_tonemap  = True
    
    ## Stage 1 of Reflectance Network Training
    ref_interval = 0
    use_sr_loss = False

    # Reflectance Network
    swin_config = get_config(opts)
    R = Net(device, opts, swin_config)
    start_iter = R.start_iter
    R_optim = R.reflectance_optim

    if start_iter > start_stage2_after:
        use_sr_loss = True

    update_sr = True if (start_iter > update_sr_after) else False

    requires_grad(R.reflectance_network.olat_encoder, True)
    requires_grad(R.reflectance_network.olat_decoder, True)
    requires_grad(R.reflectance_network.olat_sr_encoder, True)
    requires_grad(R.reflectance_network.superresolution, update_sr)

    R_ddp = DDP(R, device_ids=[rank], broadcast_buffers=False)
    R_ddp = torch.compile(R_ddp)
    R = R_ddp.module

    torch.manual_seed(rank)

    # load the dataset
    training_set_kwargs = dict(class_name='training.dataset_relight.LightstageCameraLabeledDataset', path=data, relight=relight, use_tonemap=use_tonemap,
                               ref_interval=ref_interval, use_labels=True, resolution=G.img_resolution, num_ids=num_ids, dataset_name=dataset_name, debug_mode=debug_mode,
                               lightstage_res=lightstage_res, num_training_views=num_views)
    data_loader_kwargs  = dict(pin_memory=True, prefetch_factor=None)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler  = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus, seed=0)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch//num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)

    # pbar = range(max_steps)
    # pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01, disable=False)
    from torch.distributed import get_rank, is_initialized
    
    # Check if distributed training is initialized and rank is 0
    if not is_initialized() or get_rank() == 0:
        pbar = range(max_steps)
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01, disable=False)
    else:
        # If not rank 0, disable tqdm
        pbar = range(max_steps)
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01, disable=True)
    
    if use_hdr_loss:
        calc_raw_loss    = HDRLoss(device=device)
        calc_col_loss    = nn.L1Loss()
    else:
        calc_col_loss    = nn.L1Loss()

    if lpips_wt_mode > 0:
        calc_lpips_olat_loss = VGGLoss(device=device, wt_mode=lpips_wt_mode)
    else:
        calc_lpips_olat_loss = IDMRFLoss.IDMRFLoss(device=device)
        calc_vgg_olat_loss = VGGLoss(device=device, wt_mode=2)

    # Create a dictionary dynamically from argparse options
    training_set_kwargs = {attr: getattr(opts, attr) for attr in dir(opts) if
                    not callable(getattr(opts, attr)) and not attr.startswith("__")}

    with open(f"{os.path.join(outdir, 'training_args.txt')}", "w") as file:
        dict_string = "\n".join([f"{key}: {value}" for key, value in training_set_kwargs.items()])
        file.write(dict_string)

    if (train_gen and train_real) is True:
        gen_interval = gen_interval
    else:
        gen_interval = 1

    R_reg_loss = torch.FloatTensor([0])
    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    for idx in pbar:
        i = idx + start_iter
        if i > max_steps:
            print("Finished Training")
            break

        # Stage 2
        if i == start_stage2_after:
            use_sr_loss = True
        if i == update_sr_after:
            update_sr = True
        # if i >= 120000:
        #     use_vgg_loss = True
        # else:
        #     use_vgg_loss = False

        # ----------------Training - Lightstage Data -------------------
        requires_grad(R.reflectance_network.olat_encoder, True)
        requires_grad(R.reflectance_network.olat_decoder, True)
        requires_grad(R.reflectance_network.olat_sr_encoder, update_sr)
        requires_grad(R.reflectance_network.superresolution, update_sr)

        R_loss_dict = {}; R_reg_loss_dict = {}
        R_optim.zero_grad()  # zero-out gradients

        # Sample Lightstage Data
        real_img, real_label, olat_img, dirs = next(training_set_iterator)
        
        # Multi-view
        real_img, mask = real_img[:, 0:3], real_img[:, 3:4]
        real_img = real_img.to(device).to(torch.float32) / 127.5 - 1.
        real_img_256 = face_pool(real_img)
        olat_img = olat_img.to(device).to(torch.float32) / OLAT_SCALE
        olat_img_raw = F.interpolate(olat_img, size=(128, 128), mode='bilinear', align_corners=True)
        
        mask = mask.to(device).to(torch.float16) / 255.
        mask_raw = F.interpolate(mask, size=(128, 128), mode='bilinear', align_corners=True)
        mask_raw = mask_raw.to(torch.float16)

        real_label = real_label.to(device)
        dirs = dirs.to(device)
        
        # Disabling Mask
        # olat_img = (olat_img * mask)
        # olat_img_raw = (olat_img_raw * mask_raw)
        
        # ----------------Update Reflectance Encoder and Decoder -------------------

        # Reconstruct Using Encoder
        pred_o_ls_dict, _ = R(real_img_256, real_label, real_img, dirs=dirs, relight=relight, clamp=True)
        # Mask the background
        pred_o_ls = (pred_o_ls_dict['image'] * 0.5 + 0.5)
        pred_o_ls_raw = (pred_o_ls_dict['image_raw'] * 0.5 + 0.5)
        
        # Masking
        olat_img_raw = olat_img_raw * mask_raw
        olat_img = olat_img * mask
        pred_o_ls_raw = pred_o_ls_raw * mask_raw
        pred_o_ls = pred_o_ls * mask

        ## Losses Computation
        # COLOR LOSS: Provide inputs in the range [0,1]
        loss_col = calc_col_loss(pred_o_ls_raw, olat_img_raw)
        if use_hdr_loss:
            loss_col += calc_raw_loss(pred_o_ls_raw, olat_img_raw) * 0.001

        # LPIPS LOSS: Provide inputs in the range [0,1]
        # loss_lpips = calc_lpips_olat_loss(pred_o_ls_raw, olat_img_raw)
        loss_lpips = calc_lpips_olat_loss(pred_o_ls_raw, olat_img_raw)
        # if use_vgg_loss:
        #     loss_vgg = calc_vgg_olat_loss(pred_o_ls_raw, olat_img_raw)

        if use_sr_loss:
            # COLOR LOSS
            loss_col += calc_col_loss(pred_o_ls, olat_img)
            if use_hdr_loss:
                loss_col += calc_raw_loss(pred_o_ls, olat_img) * 0.01
            
            # LPIPS LOSS
            # loss_lpips = (loss_lpips + calc_lpips_olat_loss(pred_o_ls, olat_img))*0.5
            loss_lpips = (loss_lpips + calc_lpips_olat_loss(pred_o_ls, olat_img))*0.5
            # if use_vgg_loss:
            #     loss_vgg = (loss_vgg + calc_lpips_olat_loss(pred_o_ls, olat_img))*0.5


        R_loss_dict['loss_col_olat']   = loss_col * wt_col
        R_loss_dict['loss_lpips_olat'] = loss_lpips * wt_lpips

        # if use_vgg_loss:
        #     R_loss_dict['loss_vgg_olat'] = loss_vgg * 0.2

        R_loss = sum([R_loss_dict[l] for l in R_loss_dict])
        R_loss.backward()
        R_optim.step()
        
        # Logging and Saving Results
        if get_rank() == 0:
            log_dict = {}
            desp = f' |'.join([f'{n}: {v.mean().item():.4f}' for n, v in {**R_loss_dict}.items()])
            pbar.set_description((desp))
            

            log_dict['R_loss_total'] = R_loss.detach().item()
            for key in R_loss_dict.keys():
                log_dict[key] = R_loss_dict[key].detach().item()
            
            wandb.log(log_dict)

            if i % print_interval == 0:
                os.makedirs(f'{outdir}/sample', exist_ok=True)
                with torch.no_grad():
                    disp_olat_img = olat_img.clone().detach()
                    # min_d, max_d  = pred_o_ls_dict['image_depth'].min(), pred_o_ls_dict['image_depth'].max()
                    # depth_img_mv  = ((pred_o_ls_dict['image_depth'] - min_d) / (max_d - min_d)).repeat(1, 3, 1, 1)
                    results_grid  = []
                    # results_grid.append(torch2numpy((real_img)*1, drange=255, use_tonemap=False))
                    # results_grid.append(torch2numpy((pred_o_ls_dict['image_recon'])*1, drange=255, use_tonemap=False))
                    # results_grid.append(torch2numpy((1 - depth_img_mv) * 255, drange=1))
                    
                    # OLAT GT and Predictions
                    # NOTE: torch2numpy expects data in range [-1,1]
                    results_grid.append(torch2numpy((disp_olat_img*2-1), drange=OLAT_SCALE, use_tonemap=apply_tonemap))
                    if use_sr_loss:
                        # High resolution results
                        pred_olat = (pred_o_ls_dict['image'] * 0.5 + 0.5)
                        results_grid.append(torch2numpy((pred_olat) * 2 - 1, drange=OLAT_SCALE, use_tonemap=apply_tonemap))
                    else:
                        # Low Res
                        pred_olat_raw = (pred_o_ls_dict['image_raw'] * 0.5 + 0.5)
                        results_grid.append(torch2numpy(pred_olat_raw * 2 - 1, drange=OLAT_SCALE, use_tonemap=apply_tonemap))

                    # Save Image
                    save_image_grid(results_grid, fname=f'{outdir}/sample/Real-{i:06d}.png')

                    torch.cuda.empty_cache()
            
            if i % 5000 == 0:
                os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
                snapshot_pkl = os.path.join(f'{outdir}/checkpoints/', f'network-snapshot-{i//1000:06d}.pkl')
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                snapshot_data = dict()
                snapshot_data['R'] = R_ddp.module.reflectance_network
                snapshot_data['optimizer'] = R_optim.state_dict()
                snapshot_data['start_iter'] = i
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                # torch.cuda.empty_cache()
    if rank == 0:
        wandb.finish()
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", type=str, default=None)
    # parser.add_argument("--g_ckpt", type=str, default=None)
    # parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--R_ckpt", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=500*1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_lr", type=float, default=0.002)
    parser.add_argument("--wt_tri", type=float, default=1.0)
    parser.add_argument("--wt_real", type=float, default=1.0)
    parser.add_argument("--wt_col", type=float, default=1.0)
    parser.add_argument("--wt_lpips", type=float, default=1.0)
    parser.add_argument("--wt_id", type=float, default=1.0)
    parser.add_argument("--wt_adv_mv", type=float, default=0.1)
    parser.add_argument("--wt_adv_ref", type=float, default=0.025)
    parser.add_argument("--r1_gamma", type=float, default=1.0)
    parser.add_argument("--reg_interval", type=int, default=16)
    parser.add_argument("--train_disc", type=strtobool)
    parser.add_argument("--tensorboard", type=bool, default=True)
    parser.add_argument("--print_interval", type=int, default=16)
    parser.add_argument("--start_stage2_after", type=int, default=30000)
    parser.add_argument("--update_sr_after", type=int, default=30000)
    # parser.add_argument("--start_from_latent_avg", action="store_true")
    parser.add_argument("--train_gen", type=strtobool)
    parser.add_argument("--train_real", type=strtobool)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--img_ch", type=int, default=5)
    # parser.add_argument("--outdir", type=str, default='')
    parser.add_argument("--reload", type=strtobool, default=False)
    parser.add_argument("--use_disc", type=strtobool, default=False)
    parser.add_argument("--port", type=str, default='12356')
    parser.add_argument("--gen_nrr", type=int, default=128)
    parser.add_argument("--enc_nrr", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--relight", type=strtobool)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--act_fn", type=str, default=None)
    parser.add_argument("--tonemap", type=strtobool)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--lpips_wt_mode", type=int, default=0)
    parser.add_argument("--gen_interval", type=int, default=10)
    parser.add_argument("--use_viewdirs", type=strtobool)
    parser.add_argument("--use_hdr_loss", type=strtobool)
    parser.add_argument("--eval_mode", type=strtobool)
    parser.add_argument("--debug_mode", type=strtobool)
    parser.add_argument("--num_ids", type=int, default=10)
    parser.add_argument("--lightstage_res", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=1, help="Number of training views")
    # parser.add_argument("--use_g_feat", type=strtobool)
    # GOAE Params

    ## path
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='configs/swinv2.yaml')
    parser.add_argument("--data", type=str, help='path to data directory')
    parser.add_argument("--dataset_name", type=str, help='type of dataset MPI lightstage or MERL dataset by Weyrich')
    parser.add_argument("--G_ckpt", type=str, help='path to generator model')
    parser.add_argument("--E_ckpt", type=str, help='path to GOAE encoder checkpoint')
    parser.add_argument("--AFA_ckpt", type=str, help='path to AFA model checkpoint')
    parser.add_argument("--outdir", type=str, help='path to output directory')
    # parser.add_argument("--cuda", type=str, help="specify used cuda idx ", default='0')

    ## model
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)
    parser.add_argument("--decoder_depth", type=int, default=1)
    parser.add_argument("--o_dim", type=int, default=12)
    parser.add_argument("--use_sr_encoder", type=strtobool)

    ## other
    # parser.add_argument('--batch', type=int, default=1)
    # parser.add_argument('--w_frames', type=int, default=240)
    # parser.add_argument("--multi_view", action="store_true", default=False)
    # parser.add_argument("--video", action="store_true", default=False)
    # parser.add_argument("--shape", action="store_true", default=False)
    # parser.add_argument("--edit", action="store_true", default=False)
    #
    # ## edit
    # parser.add_argument("--edit_attr", type=str, help="editing attribute direction", default="glass")
    # parser.add_argument("--alpha", type=float, help="editing alpha", default=1.0)

    opts = parser.parse_args()

    opts.port = str(int(opts.port) + randint(0, 200))

    opts.use_disc     = bool(opts.use_disc)
    opts.train_disc   = bool(opts.train_disc)
    opts.reload       = bool(opts.reload)
    opts.train_real   = bool(opts.train_real)
    opts.train_gen    = bool(opts.train_gen)
    opts.tonemap      = bool(opts.tonemap)
    opts.relight      = bool(opts.relight)
    opts.eval_mode    = bool(opts.eval_mode)
    opts.use_viewdirs = bool(opts.use_viewdirs)
    opts.use_hdr_loss = bool(opts.use_hdr_loss)
    opts.debug_mode   = bool(opts.debug_mode)
    opts.use_sr_encoder = bool(opts.use_sr_encoder)
    # opts.use_g_feat   = bool(opts.use_g_feat)

    print("Training Synthetic: ", opts.train_gen)
    print("Training Real: ", opts.train_real)
    print("Relighting", opts.relight)
    print(f"Using PORT: {opts.port}")

    if opts.R_ckpt is not None:
        ckpt_dir = os.path.dirname(opts.R_ckpt)
        ckpts = natsorted(glob.glob1(ckpt_dir, '*.pkl'))
        opts.R_ckpt = os.path.join(ckpt_dir, ckpts[-1])
        print(f'Loading {ckpts[-1]}')

    train_type = ''
    if opts.train_gen: train_type += 'gen'
    if opts.train_real: train_type += 'real'
    loss_type = 'HDR_loss' if opts.use_hdr_loss else 'L1_loss'

    desc = f'gpus{opts.num_gpus:d}-batch{opts.batch}-{opts.num_ids}ids-{opts.num_views}views-{opts.arch}-{opts.lightstage_res}_res-sr_enc_{opts.use_sr_encoder}-{opts.lpips_wt_mode}_lpips{opts.wt_lpips}-{loss_type}-lcol{opts.wt_col}-lr{opts.lr}'
    outdir = launch_training(outdir=opts.outdir, desc=desc)
    opts.outdir = outdir
    
    mp.spawn(train, args=(opts.num_gpus, opts), nprocs=opts.num_gpus, join=True)
