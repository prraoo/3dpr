# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import glob
import cv2
import math
import random

import legacy
import dlib
import time

from camera_utils import LookAtPoseSampler
# from utils.camera_utils import LookAtPoseSampler
from torch_utils import misc
import pdb
# import mrcfile

from video_uitls import convolution
from video_uitls import masking
from video_uitls import landmarks

face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("inversion/models/dlib/shape_predictor_68_face_landmarks.dat")

def crop_eyes(frame, recon_frame):
    frame = frame.astype(np.uint8)
    recon_frame = recon_frame.astype(np.uint8)
    gray = cv2.cvtColor(recon_frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        # Detect 68 facial landmark points
        lm = np.empty([1, 68, 2], dtype=int)

        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(68):
            lm[0][n][0] = face_landmarks.part(n).x
            lm[0][n][1] = face_landmarks.part(n).y

        # call mask2polypoints
        points = masking.marks2polypoints(lm)

        # mask polygons
        m = np.full_like(frame[:,:,:1], fill_value=255)
        m = masking.mask_polygons(m, points[0], gauss=75)
    return frame * (m/254.)


def get_dir2uv(light_dirs, n_lights=150):
    uv = []
    U = 10
    V = 20
    for l in range(n_lights):
        light = light_dirs[l]
        light = light.copy() / np.linalg.norm(light)

        u1 = 0.5 + math.atan2(light[0], light[2]) / (math.pi * 2)
        v1 = 1 - (0.5 + light[1] * 0.5)
        u = int(v1 * U)
        v = int(u1 * V)
        uv.append([u, v])

    return np.array(uv)
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, real_img, ws, mix_triplane, dirs, mp4=None, w_frames=60*4,
                     kind='cubic', grid_dims=(1,1),  wraps=2,  input_cam=None,
                     device=torch.device('cuda'),
                     triplane=None, gen_shapes=False, triplane_x=None, psi=1, truncation_cutoff=14, image_mode='image',
                     seeds=None, shuffle_seed=None, num_keyframes=None, cfg='FFHQ', **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    num_keyframes = 1
    pdb.set_trace()
    camera_lookat_point = torch.tensor([0, 0, 0], device=device)
    ws = ws.reshape(grid_h, grid_w, *ws.shape[1:]).unsqueeze(2).repeat(1,1,num_keyframes,1,1)


    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    render_mode = 1

    # MODE 1: fix pose and render all the olats
    if render_mode == 1:
        # Define the desired number of points after interpolation
        dirs = dirs.data.cpu().numpy()

        # Create an array of indices for interpolation
        # num_points_after_interpolation = 1500
        # indices_after_interpolation = np.linspace(0, len(dirs) - 1, num_points_after_interpolation)
        # Interpolate along the first dimension
        # dirs = np.array([np.interp(indices_after_interpolation, np.arange(len(dirs)), dirs[:, i]) for i in range(dirs.shape[1])]).T

        dirs = torch.tensor(dirs, dtype=torch.float32, device=device)
        olat_frames = len(dirs)

        # video_out = imageio.get_writer(mp4, mode='I', fps=4, codec='libx264', **video_kwargs)
        save_path = os.path.join(mp4[0], mp4[1])
        os.makedirs(save_path, exist_ok=True)
        all_poses = []
        for frame_idx in tqdm(range(num_keyframes * olat_frames)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    if input_cam is None:
                        pitch_range = 0
                        yaw_range = 0
                        cam2world_pose = LookAtPoseSampler.sample(
                            3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * olat_frames)),
                            3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * olat_frames)),
                            camera_lookat_point, radius=2.7, device=device)
                        all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    else:
                        c = input_cam

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / olat_frames)).to(device)
                    img_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane,
                                              dirs=dirs[frame_idx:frame_idx + 1])

                    olat_img = (face_pool(img_dict['image']) * 0.5 + 0.5).clamp(0, 1)
                    cv2.imwrite(os.path.join(save_path, f'{frame_idx:04d}.png'),
                                    255 * olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1])


                    # img = img_dict['image'].clamp(-1, 1).squeeze(0)
                    # img_npy = img.permute(1, 2, 0).data.cpu().numpy()[:,:,::-1]
                    # img_npy = (img_npy*0.5+0.5).clip(0,1)*255
                    #
                    # cv2.imwrite(os.path.join(save_path, f'{frame_idx:04d}.png'), img_npy)

                    # olat_img = (face_pool(rec_img_dict['image']) * 0.5 + 0.5).clamp(0, 1)
                    # olat_imgs.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
                    # if olat_idx in save_indices:
                    #     cv2.imwrite(f'{save_olat_dir}/{scan_name}_{olat_idx:03d}.png',
                    #                 255 * olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1])

                    # imgs.append(img)

            # video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        # video_out.close()

    # Mode 2: Relight with a single environment map
    if render_mode == 2:
        ## Get Environment Maps

        emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor_2018-HD/'
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/outdoor-ds/'
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor-eval/'
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/unseen-eval/'
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/unseen-eval-1/'
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor-ds-rot-new/'
        # emap_config = sorted(glob.glob1(emap_path, '*.exr'))
        # random.shuffle(emap_config)
        # emap_list = emap_config[12:13]
        emap_ids = [1, 2, 623, 686, 723, 764, 764, 986]
        emap_list = [f'EMAP-{id:04d}.exr' for id in emap_ids]
        uv_idx = get_dir2uv(dirs.data.cpu().numpy())

        os.makedirs(os.path.join(mp4[0], mp4[1]), exist_ok=True)
        # video_path = os.path.join(mp4[0],f'{mp4[1]}-natural_relit.mp4')
        # video_out = imageio.get_writer(video_path, mode='I', fps=30, codec='libx264', **video_kwargs)
        all_poses = []

        # Get all the eye masks
        tracking_frames = np.empty(shape=[num_keyframes * w_frames, 512,512,3], dtype=np.uint8)
        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            for yi in range(grid_h):
                for xi in range(grid_w):
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                              3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                              camera_lookat_point, radius=2.7, device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                    olat_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane, dirs=dirs[0:0 + 1])

                    recon_image = (olat_dict['image_recon'] * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1)*255
                    recon_image = recon_image.data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
                    tracking_frames[frame_idx] = recon_image

        print(tracking_frames.shape)
        # Get all the eye masks
        eye_masks = masking.GaussianEyeMasks(video=tracking_frames)


        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            # imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                    # loop over all the OLATs
                    olats_imgs_arr = []

                    # t_olat = time.time()

                    for olat_idx in range(len(dirs)):
                        olat_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane, dirs=dirs[olat_idx:olat_idx + 1])
                        olat_img = (olat_dict['image'] * 0.5 + 0.5).clamp(0, 1)
                        # olat_img = (face_pool(rec_img_dict['image_raw']) * 0.5 + 0.5).clamp(0, 1)
                        olats_imgs_arr.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())

                    olats_imgs_arr = np.array(olats_imgs_arr)

                    # t_olat_done = time.time() - t_olat
                    # print(f"Frame {frame_idx} took {t_olat_done} s")

                    # assert np.isnan(emap.any()) is False
                    if frame_idx == 0:
                        im_min_arr = np.zeros(shape=[len(emap_ids)], dtype=np.float32)
                        im_max_arr = np.zeros(shape=[len(emap_ids)], dtype=np.float32)

                    for emap_idx, emap_name in enumerate(emap_list):

                        save_path = os.path.join(mp4[0], mp4[1], emap_name[:-4] )
                        os.makedirs(save_path, exist_ok=True)

                        from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
                        from_emap = np.resize(from_emap, (10, 20, 3))

                        # Relighting with fixed lighting sampling
                        from_emap = from_emap[None].repeat(150, axis=0)
                        u_indices = uv_idx[:, 0].astype(int)  # First column contains u indices
                        v_indices = uv_idx[:, 1].astype(int)  # Second column contains v indices
                        from_emap = from_emap[np.arange(150), u_indices, v_indices]
                        emap = from_emap[:, None, None, :]

                        # Multiply OLATs with envmap
                        relit_image = np.sum(emap * (olats_imgs_arr ** 2), axis=0)
                        if frame_idx == 0:
                            im_min_arr[emap_idx] = relit_image.min()
                            im_max_arr[emap_idx] = relit_image.max()

                        relit_image = (relit_image - im_min_arr[emap_idx]) / (im_max_arr[emap_idx] - im_min_arr[emap_idx])
                        relit_image = np.sqrt(relit_image)
                        relit_image_8bit = relit_image.clip(0, 1) * 255

                        # Create the 20x40 image to be added
                        scale = 8
                        emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))[:, :, ::-1]
                        image_20x40 = cv2.resize(emap_png, (20 * scale, 10 * scale), interpolation=cv2.INTER_AREA)

                        # Determine the position to place the 20x40 image in the lower right corner
                        x_offset = 512 - 20 * scale  # Calculate the x-offset for the lower right corner
                        y_offset = 512 - 10 * scale  # Calculate the y-offset for the lower right corner

                        # Copy the 20x40 image into the lower right corner of the 512x512 image
                        relit_image_8bit[y_offset:y_offset + 10 * scale, x_offset:x_offset + 20 * scale] = image_20x40

                        # mask eyes
                        relit_image_8bit = eye_masks.mask_frame(frame_idx, relit_image_8bit)
                        cv2.imwrite(os.path.join(save_path, f'{frame_idx:04d}_{mp4[1]}.png'), relit_image_8bit[:, :, ::-1])

                        # relit_image = torch.tensor(relit_image_8bit.transpose(2,0,1), dtype=torch.float32)/255 * 2 - 1

                        # imgs.append(relit_image)


        #     video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        # video_out.close()

    #Mode 3: rotating env maps sequence within the inline code
    if render_mode == 3:
        # emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor_2018-HD/'
        emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/unseen-eval/'
        emap_config = sorted(glob.glob1(emap_path, '*.exr'))
        # random.shuffle(emap_config)
        emap_list = emap_config
        # uv_idx = get_dir2uv(dirs.data.cpu().numpy())

        light_dirs = np.load('/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy')
        light_dirs_npy = light_dirs / np.linalg.norm(light_dirs, axis=1)[..., None]
        uv_idx = get_dir2uv(light_dirs_npy)

        video_path = os.path.join(mp4[0], f'{mp4[1]}-rot-relit.mp4')
        video_out = imageio.get_writer(video_path, mode='I', fps=2, codec='libx264', **video_kwargs)
        all_poses = []
        olat_img = None
        for frame_idx in tqdm(range(num_keyframes * 40)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    # pitch_range = 0.0
                    # yaw_range = 0.0
                    # cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    #                                           3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    #                                           camera_lookat_point, radius=2.7, device=device)

                    pitch = 0.0
                    yaw = -0.25
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw, 3.14/2 -0.05 + pitch,
                                                              camera_lookat_point, radius=2.7, device=device)

                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                    # loop over all the OLATs
                    if olat_img is None:
                        olats_imgs_arr = []
                        for olat_idx in range(len(dirs)):
                            olat_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane, dirs=dirs[olat_idx:olat_idx + 1])
                            olat_img = (olat_dict['image'] * 0.5 + 0.5).clamp(0, 1)
                            olats_imgs_arr.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())

                        olats_imgs_arr = np.array(olats_imgs_arr)

                    # Modify enviroment maps
                    emap_name = emap_list[9]
                    from_emap_raw = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
                    from_emap_raw = np.roll(from_emap_raw, frame_idx, axis=1)
                    from_emap = np.resize(from_emap_raw, (10, 20, 3))

                    # Relighting with fixed lighting sampling
                    from_emap = from_emap[None].repeat(150, axis=0)
                    u_indices = uv_idx[:, 0].astype(int)  # First column contains u indices
                    v_indices = uv_idx[:, 1].astype(int)  # Second column contains v indices
                    from_emap = from_emap[np.arange(150), u_indices, v_indices]
                    emap = from_emap[:, None, None, :]

                    assert np.isnan(emap.any()) == False
                    # Multiply OLATs with envmap
                    relit_image = np.sum(emap * (olats_imgs_arr ** 2), axis=0)
                    relit_image = (relit_image - relit_image.min()) / (relit_image.max() - relit_image.min())
                    relit_image = np.sqrt(relit_image)
                    relit_image_8bit = relit_image.clip(0, 1) * 255

                    # Create the 20x40 image to be added
                    scale = 8
                    emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))[:, :, ::-1]
                    emap_png = np.roll(emap_png, frame_idx, axis=1)
                    image_20x40 = cv2.resize(emap_png, (20 * scale, 10 * scale), interpolation=cv2.INTER_AREA)

                    # Determine the position to place the 20x40 image in the lower right corner
                    x_offset = 512 - 20 * scale  # Calculate the x-offset for the lower right corner
                    y_offset = 512 - 10 * scale  # Calculate the y-offset for the lower right corner

                    # Copy the 20x40 image into the lower right corner of the 512x512 image
                    relit_image_8bit[y_offset:y_offset + 10 * scale, x_offset:x_offset + 20 * scale] = image_20x40
                    relit_image = torch.tensor(relit_image_8bit.transpose(2,0,1), dtype=torch.float32)/255 * 2 - 1
                    imgs.append(relit_image)


            video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        video_out.close()

    #Mode 4: rotating env maps sequence within the inline code
    if render_mode == 4:
        # view-dependent effects - vary pitch and render
        # video_path = os.path.join(mp4[0], f'{mp4[1]}-vd-effects.mp4')
        # video_out = imageio.get_writer(video_path, mode='I', fps=20, codec='libx264', **video_kwargs)

        dir_idx = 149
        save_path = os.path.join(mp4[0], mp4[1], f'{dir_idx:03d}')
        os.makedirs(save_path, exist_ok=True)
        all_poses = []

        # Get all the eye masks
        tracking_frames = np.empty(shape=[num_keyframes * w_frames, 512,512,3], dtype=np.uint8)
        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            for yi in range(grid_h):
                for xi in range(grid_w):
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                              3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                              camera_lookat_point, radius=2.7, device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                    olat_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane, dirs=dirs[0:0 + 1])

                    recon_image = (olat_dict['image_recon'] * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1)*255
                    recon_image = recon_image.data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
                    tracking_frames[frame_idx] = recon_image

        print(tracking_frames.shape)
        # Get all the eye masks
        eye_masks = masking.GaussianEyeMasks(video=tracking_frames)

        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0.00 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                    img_dict = G.forward_eval(real_img, rec_ws=w.unsqueeze(0), c=c, mix_triplane=mix_triplane, dirs=dirs[dir_idx: dir_idx + 1])
                    img = img_dict['image'].clamp(-1, 1).squeeze(0)

                    img_npy = img.permute(1, 2, 0).data.cpu().numpy()[:,:,::-1]
                    img_npy = (img_npy*0.5+0.5).clip(0,1)*255
                    # img_npy = eye_masks.mask_frame(frame_idx, img_npy)
                    cv2.imwrite(os.path.join(save_path, f'{frame_idx:04d}.png'), img_npy.astype(np.uint8))

                    # imgs.append(img)


        #     video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        # video_out.close()


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
        # m = range_re.match(p)
        # if m :
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
    # m = re.match(r'^(\d+)[x,](\d+)$', s)
    # if m :
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if interpolate:
        output = os.path.join(outdir, 'interpolation.mp4')
        gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
