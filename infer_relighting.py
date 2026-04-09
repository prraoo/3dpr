import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import datetime
import dnnlib
import numpy as np
import torch
import torchvision
from configs.infer_config import get_parser
from configs.swin_config import get_config
from training.networks import Net
from tqdm import tqdm
from camera_utils import LookAtPoseSampler
import glob
import cv2
from natsort import natsorted
import random
from image_utils.infer_mask import generate_rmbg_mask
import math
from gen_shape import gen_mesh
from image_utils.data_util import resize_n_crop_numpy
from image_utils.tonemap import apply_tonemap as process_image
from image_utils.general import add_border

np.random.seed(42)
random.seed(108)


def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
	
	if yaw is None:
		yaw = np.random.uniform(-yaw_range, yaw_range)
	if pitch is None:
		pitch = np.random.uniform(-pitch_range, pitch_range)
	
	cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
	return c


def get_pose_video(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
	if yaw is None:
		yaw = np.random.uniform(-yaw_range, yaw_range)
	if pitch is None:
		pitch = np.random.uniform(-pitch_range, pitch_range)
	
	cam2world_pose = LookAtPoseSampler.sample(yaw, pitch, cam_pivot, radius=cam_radius, device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1, -1)
	return c


def build_dataloader(data_path, batch=1, use_mask=False, pin_memory=True, prefetch_factor=2):
	dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_face.CameraLabeledDataset', path=data_path,
	                                 use_labels=True, max_size=None, xflip=False,resolution=256, use_512 = True, use_mask=use_mask)
	dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
	dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, pin_memory=pin_memory, prefetch_factor=prefetch_factor, num_workers=4)
	# dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, pin_memory=pin_memory, prefetch_factor=None, num_workers=0)
	
	return dataloader, dataset


def get_lightdirs(device, dataset_name, light_dirs_path=None):
	if dataset_name == 'mpi':
		light_dirs_path = light_dirs_path or '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/LSX_light_positions_mpi.txt'
		light_dirs = np.loadtxt(light_dirs_path).astype(np.float32)
	elif dataset_name == 'weyrich':
		light_dirs_path = light_dirs_path or '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy'
		light_dirs = np.load(light_dirs_path).astype(np.float32)
	else:
		raise NotImplementedError
	return torch.FloatTensor(light_dirs / np.linalg.norm(light_dirs, axis=1)[..., None]).to(device)


def get_dir2uv(light_dirs, uv_res, dataset_name):
	# uv = []
	# U = uv_res[0]
	# V = uv_res[1]
	#
	# for l in range(len(light_dirs)):
	# 	light = light_dirs[l]
	# 	u1 = 0.5 + math.atan2(light[0], light[2]) / (math.pi * 2)
	# 	v1 = 1 - (0.5 + light[1] * 0.5)
	# 	u = int(np.clip(v1 * U, 0, U-1))
	# 	v = int(np.clip(u1 * V, 0, V-1))
	# 	uv.append([u, v])
	#
	# return np.array(uv)
	
	# directions = light_dirs.copy()
	# norms = np.linalg.norm(directions, axis=1, keepdims=True)
	# Convert to spherical coordinates
	if dataset_name == 'mpi':
		unit_directions = light_dirs.copy()
		height = uv_res[0]
		width = uv_res[1]
		back_idx = 74
		phi_front = np.arctan2(unit_directions[0:back_idx, 0], unit_directions[0:back_idx, 1])  # Azimuthal angle
		theta_front = np.arccos(-unit_directions[0:back_idx, 2])  # Polar angle
		
		phi_back = np.arctan2(unit_directions[back_idx:, 0], unit_directions[back_idx:, 1])  # Azimuthal angle
		theta_back = np.arccos(-unit_directions[back_idx:, 2])  # Polar angle
		
		phi = np.concatenate([phi_front, phi_back])
		theta = np.concatenate([theta_front, theta_back])
		
		# Map to environment map coordinates
		x = ((phi + np.pi) / (2 * np.pi) * width).astype(int) % width
		y = ((theta / np.pi) * height).astype(int) % height
		return np.stack((y, x), axis=-1)
	elif dataset_name == 'weyrich':
		uv = []
		U = 10
		V = 20
		for l in range(len(light_dirs)):
			light = light_dirs[l]
			light = light.copy() / np.linalg.norm(light)
			
			u1 = 0.5 + math.atan2(light[0], light[2]) / (math.pi * 2)
			v1 = 1 - (0.5 + light[1] * 0.5)
			u = int(np.clip(v1 * U, 0, U - 1))
			v = int(np.clip(u1 * V, 0, V - 1))
			uv.append([u, v])
		return np.array(uv)
	else:
		raise NotImplementedError

	

def get_envmap_mask(zspiral_indices, n_olats, opts):
	mask_array = []
	for olat_idx in range(n_olats):
		mask = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{zspiral_indices[olat_idx]:03d}.png'))
		mask = cv2.flip(mask, 1) / 255
		mask_array.append(mask)
	return np.array(mask_array)


def add_inset_envmaps(zspiral_indices, olat_indices, opts):
	mask_array = []
	for idx in olat_indices:
		mask = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{zspiral_indices[idx]:03d}.png'))
		mask = cv2.flip(mask, 1) / 255
		mask_array.append(mask)
	return np.array(mask_array).sum(0) * 255


def sample_uv_from_envmap(envmap, mask_array, fn='max'):
	emap_res = (256, 512)
	envmap = cv2.resize(envmap, (emap_res[1], emap_res[0]))
	envmap = envmap[None, ...]
	envmap_sampled = envmap * mask_array
	if fn == 'max':
		return envmap_sampled.max((1,2))
	elif fn == 'mean':
		return envmap_sampled.mean((1,2))
	else:
		raise NotImplementedError
		
	#
	#
	# for olat_idx in range(n_olats):
	# 	mask = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{zsprial_indices[olat_idx]:03d}.png'))
	# 	mask = cv2.flip(mask, 1)
	# 	mask_array.append(mask)
	# 	max_value = (envmap * mask).max(axis=(0, 1))
	# 	envmap_sampled.append(max_value)
	#
	# import pdb; pdb.set_trace()
	#
	#
	# return np.array(envmap_sampled)


def load_olat_image(fname, params):
	img = cv2.imread(fname, -1)
	img = process_image(img)
	img = (img / 65535) * 255
	img = resize_n_crop_numpy(img, params)
	return img


# combined_olat = combine_olat_images(gt_olat_dir, gt_fnames_list, save_indices, scale_crop_params)


def combine_olat_images(olat_path, olat_fnames_list, indices, params):
	olat_list = []
	for idx in indices:
		olat = load_olat_image(os.path.join(olat_path, olat_fnames_list[idx]), params) / 255
		olat = olat ** 2
		olat_list.append(olat)
	combined_olat = np.array(olat_list).sum(0)
	combined_olat = np.sqrt(combined_olat) * 255
	
	return combined_olat.astype(np.uint8)

@torch.no_grad()
def infer_main(opts, device, now):
	gen_shape = opts.shape
	use_gt_olat = False
	
	emap_path = opts.envmap_dir or '/HPS/prao2/static00/datasets/Environment-Maps/quant-eval/'

	emap_config = natsorted(glob.glob1(emap_path, '*.exr'), reverse=False)
	# random.shuffle(emap_config)
	emap_list = emap_config[:30]
	
	## camera parameters
	cam_pivot = torch.tensor([0, 0, 0.2], device=device)
	intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device=device)
	face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))
	
	## build model
	swin_config = get_config(opts)
	net = Net(device, opts, swin_config)
	net.eval()
	
	
	## build data
	dataloader, dataset = build_dataloader(data_path=opts.data, batch=opts.batch, use_mask=opts.use_mask)
	
	## get light dirs
	dirs = get_lightdirs(device, dataset_name=opts.dataset_name, light_dirs_path=opts.light_dirs_path)

	if opts.dataset_name == 'mpi':
		UV_res = [13 * 1, 26 * 1]  # 14*28=392
		UV_inset_res = [13 * 1, 26 * 1]  # a white background
		
		ls_freq = 1 if opts.lightstage_res == 'full' else 2
		
		full_light_indices = [0, 20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230, 251, 272, 293, 314, 335, 348, 349]
		bad_light_indices = [270, 296, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
							 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]
		
		all_light_indices = [x for x in range(len(dirs)) if (x not in full_light_indices and x < 350)]
		all_light_indices = [x for x in all_light_indices if x not in bad_light_indices]
		
		
		# Inset envmap list
		inset_envmap_fnames = np.array([i for i in range(350)], dtype=np.int16)
		
		# Sample Needed Lights
		dirs = dirs[all_light_indices]
		select_indices = [idx for idx, _ in enumerate(dirs) if idx % ls_freq == 0]
		dirs = dirs[select_indices]  # final list of lights
		
		# Select needed inset envmaps
		inset_envmap_fnames = inset_envmap_fnames[all_light_indices]
		inset_envmap_fnames = inset_envmap_fnames[select_indices]
		n_olats = len(dirs)
		assert len(inset_envmap_fnames) == n_olats, 'Envmaps do not match lights'
		
	elif opts.dataset_name == 'weyrich':
		UV_res = [10, 20]
		UV_inset_res = [13 * 1, 26 * 1]  # a white background
		all_light_indices = [x for x in range(len(dirs))]
		ls_freq = 1
		n_olats = len(dirs)

	
	
	# Get the environment pixels to be sampled
	light_dirs_npy = dirs.data.cpu().numpy()
	uv_idx = get_dir2uv(light_dirs_npy, uv_res=UV_res, dataset_name=opts.dataset_name)
	
	## main loop
	num_images = 0
	for data in tqdm(dataloader, disable=True):
		num_images += 1
		if num_images == 2:
			print(f'Loading {img_name}')
			# raise AssertionError
		real_img, real_label, real_img_512, img_name = data
		
		real_img = real_img.to(device).to(torch.float32) / 127.5 - 1.
		real_label = real_label.to(device)
		real_img_512 = real_img_512.to(device).to(torch.float32) / 127.5 - 1.
		
		reference_img = real_img_512.data.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, ::-1]
		reference_img = (reference_img*0.5+0.5)*255
		
		reference_mask = generate_rmbg_mask(reference_img)
		ref_m = reference_mask / 255.
		
		if not opts.input_view:
			rec_img_dict, rec_img_dict_w = net(real_img, real_label, real_img_512, dirs=dirs[0:0+1], return_latents=True)
			mix_triplane = rec_img_dict['mix_triplane']
			rec_ws = rec_img_dict['rec_ws']
			scan_name = img_name[0].split(".")[0].split('_')[1]
		else:
			olat_imgs = []
			scan_name = img_name[0].split(".")[0].split('_')[1]
			
			save_invert_dir = os.path.join(opts.outdir, 'invert', scan_name)
			os.makedirs(save_invert_dir, exist_ok=True)
			
			save_dir = os.path.join(opts.outdir, 'relit', scan_name)
			# save_dir = os.path.join(opts.outdir, 'relit')
			os.makedirs(save_dir, exist_ok=True)
			
			save_olat_dir = os.path.join(opts.outdir, 'olat', scan_name)
			os.makedirs(save_olat_dir, exist_ok=True)
			
			# save_olat_emap_dir = os.path.join(opts.outdir, 'olat_emap', scan_name)
			# os.makedirs(save_olat_emap_dir, exist_ok=True)

			save_indices = np.arange(0, n_olats).tolist()
			# save_indices = [0]
			# save_indices = [81, 141]
			# save_indices = [81, 103]
			# save_indices = [78, 99]
			# save_indices = [3, 36, 78, 86, 99]
			# save_indices = [14, 16, 26, 54, 74, 87]
			
			# save_indices = []
			
			# Determine the position to place the 20x40 image in the lower right corner
			scale = 6
			bs = 1  # border size
			# WARNING: hardcoded
			x_offset = 512 - UV_inset_res[1] * scale  # Calculate the x-offset for the lower right corner
			y_offset = 512 - UV_inset_res[0] * scale  # Calculate the y-offset for the lower right corner
			# olat_emap_full = np.ones((UV_inset_res[0], UV_inset_res[1]), dtype=np.uint8) * 255  # background with a white outline
			inset_olat_list = []
			# olat_stencil = np.ones((14, 28), dtype=np.uint8)*255
			for olat_idx in range(n_olats):
				
				if opts.dataset_name == 'mpi':
					# olat_emap = np.zeros((UV_res[0], UV_res[1]), dtype=np.uint8)
					# olat_emap[uv_idx[olat_idx][0], uv_idx[olat_idx][1]] = 255
					# olat_emap_full[:, :] = olat_emap
					
					olat_emap = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{inset_envmap_fnames[olat_idx]:03d}.png'))
					olat_emap = cv2.flip(olat_emap, 1)

				# if olat_idx < 148:
				# 	continue
				
				# cv2.imwrite(os.path.join(save_olat_emap_dir, f'{olat_idx:03d}.exr'), olat_emap.astype(np.float32))
				rec_img_dict, rec_img_dict_w = net(real_img, real_label, real_img_512, dirs=dirs[olat_idx:olat_idx+1], return_latents=True)
				
				if olat_idx == 0:
					invert_img = (face_pool(rec_img_dict['image_recon']) * 0.5 + 0.5).clamp(0, 1)
					np_img = 255 * invert_img.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1]
					cv2.imwrite(f'{save_invert_dir}/{scan_name}_invert.png', np_img * ref_m)

				if opts.shape:
					gen_mesh(net.decoder, rec_img_dict['rec_ws'], rec_img_dict['mix_triplane'], save_invert_dir)

				mix_triplane = rec_img_dict['mix_triplane']
				rec_ws = rec_img_dict['rec_ws']
				olat_img = (rec_img_dict['image'] * 0.5 + 0.5).clamp(0, 1)
				olat_imgs.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
				if olat_idx in save_indices:
					np_img = 255 * olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1] * ref_m
					# # TODO: add olat stencil as an inset image
					if opts.dataset_name == 'mpi':
						inset_emap = cv2.resize(olat_emap, (UV_inset_res[1]*scale, UV_inset_res[0]*scale), interpolation=cv2.INTER_AREA)
						inset_emap = add_border(inset_emap, border_size=bs)
						# np_img[y_offset:y_offset + UV_inset_res[0]*scale, x_offset:x_offset + UV_inset_res[1]*scale] = inset_emap[:, :, None].repeat(3, axis=2)
						# np_img[y_offset-2*bs:y_offset + UV_inset_res[0]*scale, x_offset-2*bs:x_offset + UV_inset_res[1]*scale] = inset_emap
						inset_olat_list.append(inset_emap)
					
					if use_gt_olat and (opts.dataset_name == 'mpi'):
						gt_img = load_olat_image(os.path.join(gt_olat_dir, gt_fnames_list[olat_idx]), scale_crop_params)
						# gt_img[y_offset:y_offset + UV_inset_res[0] * scale, x_offset:x_offset + UV_inset_res[1] * scale] = inset_emap[:, :, None].repeat(3, axis=2)
						gt_img[y_offset-2*bs:y_offset + UV_inset_res[0] * scale, x_offset-2*bs:x_offset + UV_inset_res[1] * scale] = inset_emap
						# np_img = cv2.hconcat([gt_img, np_img])
						# np_img = cv2.hconcat([gt_img])
						np_img = cv2.hconcat([np_img])
					cv2.imwrite(f'{save_olat_dir}/{scan_name}_{olat_idx:03d}.png', np_img)
			
			# Load Random Envmaps and Relight
			# Gather All OLATs
			olat_imgs_arr = np.array(olat_imgs)
			if opts.dataset_name == 'mpi':
				emap_mask_array = get_envmap_mask(inset_envmap_fnames, n_olats, opts)
			
			# if opts.add_envmap:
			# 	assert len(save_indices) <= 3, "adding more than 3 envmaps does not makes sense!"
			# 	inset_emap = add_inset_envmaps(inset_envmap_fnames, save_indices, opts)
			# 	breakpoint()
			# 	emap = np.zeros((n_olats, 3), dtype=np.uint8)
			# 	emap[save_indices] = 1
			#
			# 	relit_image = np.sum(emap[:, None, None, :] * (olat_imgs_arr ** 2), axis=0)
			# 	relit_image = np.sqrt(relit_image)
			# 	relit_image_8bit = relit_image.clip(0, 1) * 255
			#
			# 	# Create the 20x40 image to be added
			# 	inset_emap = cv2.resize(inset_emap, (UV_inset_res[1] * scale, UV_inset_res[0] * scale), interpolation=cv2.INTER_AREA)
			# 	inset_emap = add_border(inset_emap, border_size=bs)
			#
			# 	# Copy the inset envmap into the lower right corner of the 512x512 image
			# 	np_img = ((relit_image_8bit * ref_m)[:, :, ::-1]).astype(np.uint8)
			# 	np_img[y_offset-2*bs:y_offset + UV_inset_res[0] * scale, x_offset-2*bs:x_offset + UV_inset_res[1] * scale] = inset_emap
			# 	prefix = ''.join(f'{x}_' for x in save_indices)
			# 	if use_gt_olat:
			# 		gt_img = combine_olat_images(gt_olat_dir, gt_fnames_list, save_indices, scale_crop_params)
			# 		gt_img[y_offset - 2 * bs:y_offset + UV_inset_res[0] * scale, x_offset - 2 * bs:x_offset + UV_inset_res[1] * scale] = inset_emap
			# 		np_img = cv2.hconcat([gt_img, np_img])
			# 	cv2.imwrite(f'{save_olat_dir}/combined_{prefix}{scan_name}.png', np_img)
			
			for ii, emap_name in enumerate(emap_list):
				from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
				# from_emap = cv2.resize(from_emap, (UV_res[1]*3, UV_res[0]*3))
				# from_emap = cv2.resize(from_emap, (UV_res[1]+1, UV_res[0]+1))
				# from_emap = (from_emap - from_emap.min()) / (from_emap.max() - from_emap.min())
				
				# # White Light
				# if ii > 0:
				# 	from_emap = from_emap.max(axis=2, keepdims=True)
				# 	from_emap = np.where(from_emap > 0.60, 1, 0)
				# 	from_emap_mask = from_emap.copy().astype(np.uint8)
				# else:
				# 	from_emap = np.where(from_emap > 0.0, 1, 0)

				
				# # Relighting with incorrect uv sampling
				# from_emap = from_emap.reshape(-1, 3)
				# emap = np.tile(from_emap, (512, 512, 1, 1)).transpose(2, 0, 1, 3)
				# emap = emap[uv_idx]
				
				
				# # Relighting with fixed lighting sampling
				# from_emap = from_emap[None].repeat(n_olats, axis=0)
				# u_indices = uv_idx[:, 0].astype(int)  # First column contains u indices
				# v_indices = uv_idx[:, 1].astype(int)  # Second column contains v indices
				# emap = from_emap[np.arange(n_olats), u_indices, v_indices]
				
				# Relighting with sampled envmap masks
				from_emap = np.resize(from_emap, (UV_res[0], UV_res[1], 3))
				emap = sample_uv_from_envmap(from_emap, emap_mask_array, fn=opts.emap_sample_fn)
				
				if not np.isnan(emap.any()):
					# Multiply OLATs with envmap
					# relit_image = np.mean(emap * (olat_imgs_arr**2), axis=0)
					relit_image = np.sum(emap[:, None, None, :] * (olat_imgs_arr**2), axis=0)
					im_max = relit_image.max()
					im_min = relit_image.min()
					relit_image = (relit_image - relit_image.min()) / (relit_image.max() - relit_image.min())
					relit_image = np.sqrt(relit_image)
					relit_image_8bit = relit_image.clip(0, 1) * 255
					
					# Create the 20x40 image to be added
					# scale = 8
					emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))
					emap_inset = cv2.resize(emap_png, (UV_inset_res[1]*scale, UV_inset_res[0]*scale), interpolation=cv2.INTER_AREA)
					
					# # White Light
					# if ii > 0:
					# 	emap_inset = emap_inset.max(axis=2, keepdims=True)
					# 	emap_inset = np.where(emap_inset > 255 * 0.6, 1 * 255, 0)
					# else:
					# 	emap_inset = np.where(emap_inset > 0, 1 * 255, 0)

					
					# Determine the position to place the 20x40 image in the lower right corner
					x_offset = 512 - UV_inset_res[1]*scale  # Calculate the x-offset for the lower right corner
					y_offset = 512 - UV_inset_res[0]*scale  # Calculate the y-offset for the lower right corner
					
					# Copy the 20x40 image into the lower right corner of the 512x512 image
					
					np_img = ((relit_image_8bit * ref_m)[:,:,::-1]).astype(np.uint8)
					# np_img[y_offset:y_offset + UV_inset_res[0]*scale, x_offset:x_offset + UV_inset_res[1]*scale] = emap_inset
					# cv2.imwrite(f'{save_dir}/{scan_name}_{emap_name[:-4]}.png', cv2.hconcat([reference_img.astype(np.uint8), np_img]))
					cv2.imwrite(f'{save_dir}/{scan_name}_{emap_name[:-4]}.png', cv2.hconcat([np_img]))
		
		if opts.multi_view:
			emap_config = natsorted(glob.glob1(emap_path, '*.exr'))
			emap_list = emap_config[:45]
			
			assert len(emap_list) > 0
			
			save_invert_dir = os.path.join(opts.outdir, 'invert_mv', scan_name)
			os.makedirs(save_invert_dir, exist_ok=True)
			
			save_invert_mask_dir = os.path.join(opts.outdir, 'mask_mv', scan_name)
			os.makedirs(save_invert_mask_dir, exist_ok=True)
			
			coef = [3*x/30 for x in range(-30, 30)]
			coef_pitch = [0 for x in range(-30, 30)]
			n_views = len(coef)

			yaw_list = []
			pitch_list = []
			
			for cam_idx in range(len(coef)):
				yaw_list.append(coef[cam_idx] * np.pi * 25 / 360)
				pitch_list.append(coef_pitch[cam_idx] * np.pi * 20 / 360)
			
			masks_mv_list = []
			if opts.gen_mask_mode == 1:
				for cam_idx in range(len(coef)):
					# yaw = coef[cam_idx] * np.pi * 25 / 360
					# pitch = coef_pitch[cam_idx] * np.pi * 20 / 360
					c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw_list[cam_idx], pitch=pitch_list[cam_idx])
					
					olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane, dirs=dirs[0:1])
					invert_img_novel_view = (face_pool(olat_dict_novel_view['image_recon']) * 0.5 + 0.5).clamp(0, 1)
					save_image = 255 * invert_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1]
					cv2.imwrite(f'{save_invert_dir}/{cam_idx:03d}.jpg', 255 * invert_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1])
				return 0
			
			elif opts.gen_mask_mode == 2:
				for cam_idx in range(len(coef)):
					c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw_list[cam_idx], pitch=pitch_list[cam_idx])
					olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane, dirs=dirs[0:1])
					invert_img_novel_view = (face_pool(olat_dict_novel_view['image_recon']) * 0.5 + 0.5).clamp(0, 1)
					save_image = 255 * invert_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()
					
					# from image_utils.infer_mask import generate_rmbg_mask
					masks_mv_list.append(generate_rmbg_mask(save_image))
				
			
			save_dir = os.path.join(opts.outdir, 'relit_mv', scan_name)
			os.makedirs(save_dir, exist_ok=True)
			
			save_olat_dir = os.path.join(opts.outdir, 'olat_mv', scan_name)
			os.makedirs(save_olat_dir, exist_ok=True)
			
			all_olats_list = []
			# save_indices = np.random.randint(0, 151, size=50)
			opts.use_mask = True
			if opts.use_mask and opts.gen_mask_mode == 1:
				mask_images = natsorted(glob.glob(os.path.join(save_invert_mask_dir, '*.png')))
				assert len(mask_images) == len(coef), f'Number of masks are not equal to rendered views'
			elif opts.use_mask and opts.gen_mask_mode == 2:
				assert len(masks_mv_list) == len(coef), f'Number of masks are not equal to rendered views'
				
			
			for olat_idx in range(n_olats):
				olats_list = []
				
				for cam_idx in range(len(coef)):
					yaw = coef[cam_idx] * np.pi*25/360
					pitch = coef_pitch[cam_idx] * np.pi*20/360
					c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					
					olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane, dirs=dirs[olat_idx:olat_idx+1])
					
					olat_img_novel_view = face_pool(olat_dict_novel_view['image'])
					# olat_img_novel_view = face_pool(olat_dict_novel_view['image_raw'])
					olat_img_novel_view = (olat_img_novel_view * 0.5 + 0.5).clamp(0, 1)
					olat_img_novel_view = olat_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()
					olats_list.append(olat_img_novel_view)
					
					if olat_idx in save_indices:
						idx = save_indices.index(olat_idx)
						inset_emap = inset_olat_list[idx]
						# cv2.imwrite(f'{save_olat_dir}/{scan_name}_Cam{cam_idx:02d}_{olat_idx:03d}.png', 255*olat_img_novel_view[:, :, ::-1]*mv_masks[cam_idx])
						if opts.use_mask and opts.gen_mask_mode == 1:
							mask = cv2.imread(mask_images[cam_idx])/255.
							final_olat_image = olat_img_novel_view[:, :, ::-1] * mask * 255
						elif opts.use_mask and opts.gen_mask_mode == 2:
							mask = masks_mv_list[cam_idx] / 255.
							final_olat_image = olat_img_novel_view[:, :, ::-1] * mask * 255
						else:
							final_olat_image = olat_img_novel_view[:, :, ::-1] * 255
						# final_olat_image[y_offset - 2 * bs:y_offset + UV_inset_res[0] * scale,x_offset - 2 * bs:x_offset + UV_inset_res[1] * scale] = inset_emap
						cv2.imwrite(f'{save_olat_dir}/{scan_name}_Cam{cam_idx:02d}_{olat_idx:03d}.png', final_olat_image)
				
				all_olats_list.append(olats_list)
			
			all_olats_arr = np.array(all_olats_list).transpose(1, 0, 2, 3, 4)
			
			# Relighting Novel Views
			for emap_name in emap_list:
				print(f'Relighting with {emap_name}')
				from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
				# from_emap = (from_emap - from_emap.min()) / (from_emap.max() - from_emap.min())
				# from_emap = np.resize(from_emap, (14, 28, 3))
				
				# # Relighting with incorrect uv sampling
				# from_emap = from_emap.reshape(-1, 3)
				# emap = np.tile(from_emap, (512, 512, 1, 1)).transpose(2, 0, 1, 3)
				# emap = emap[uv_idx]
				
				# # Relighting with fixed lighting sampling
				# from_emap = from_emap[None].repeat(n_olats, axis=0)
				# u_indices = uv_idx[:, 0].astype(int)  # First column contains u indices
				# v_indices = uv_idx[:, 1].astype(int)  # Second column contains v indices
				# from_emap = from_emap[np.arange(n_olats), u_indices, v_indices]
				
				
				# Relighting with sampled envmap masks
				from_emap = np.resize(from_emap, (UV_res[0], UV_res[1], 3))
				emap = sample_uv_from_envmap(from_emap, emap_mask_array, fn=opts.emap_sample_fn)

				if not np.isnan(emap.any()):
					for view_idx in range(n_views):
						# Multiply OLATs with envmap
						relit_img_arr = np.sum(emap[:, None, None, :] * (all_olats_arr[view_idx] ** 2), axis=0)
						relit_image = np.sqrt(relit_img_arr)
						if view_idx == 0:
							im_max = relit_image.max()
							im_min = relit_image.min()
						relit_image_8bit = (relit_image - im_min) / (im_max - im_min)
						relit_image_8bit = relit_image_8bit.clip(0, 1) * 255
						
						# Create the 20x40 image to be added
						scale = 8
						emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))
						image_20x40 = cv2.resize(emap_png, (20 * scale, 10 * scale), interpolation=cv2.INTER_AREA)
						image_20x40 = add_border(image_20x40, border_size=bs)
						
						# Determine the position to place the 20x40 image in the lower right corner
						x_offset = 512 - 20 * scale  # Calculate the x-offset for the lower right corner
						y_offset = 512 - 10 * scale  # Calculate the y-offset for the lower right corner
						
						# Copy the 20x40 image into the lower right corner of the 512x512 image
						
						# WARNING: Tonemapping missing before saving
						
						# cv2.imwrite(f'{save_dir}/{scan_name}_{emap_name[:-4]}_view{view_idx:03d}.png', cv2.hconcat([reference_img, relit_image_8bit[:, :, ::-1]]))
						# cv2.imwrite(f'{save_dir}/{scan_name}_{emap_name[:-4]}_view{view_idx:03d}.png', cv2.hconcat([relit_image_8bit[:, :, ::-1]*mv_masks[cam_idx]]))
						if opts.use_mask and opts.gen_mask_mode==1:
							mask = cv2.imread(mask_images[view_idx])/255.
							final_relit_image = relit_image_8bit[:, :, ::-1] * mask
						elif opts.use_mask and opts.gen_mask_mode == 2:
							mask = masks_mv_list[view_idx] / 255.
							final_relit_image = relit_image_8bit[:, :, ::-1] * mask
						else:
							final_relit_image = olat_img_novel_view[:, :, ::-1]
						# final_relit_image[y_offset-2*bs:y_offset + 10 * scale, x_offset-2*bs:x_offset + 20 * scale] = image_20x40
						cv2.imwrite(f'{save_dir}/{scan_name}_{emap_name[:-4]}_view{view_idx:03d}.png', final_relit_image)
		
		
		if opts.video:
			import imageio
			from gen_video import gen_interp_video
			from gen_shape import gen_mesh
			# camera_lookat_point = torch.tensor(net.decoder.rendering_kwargs['avg_camera_pivot'], device=device)
			
			render_mode = opts.render_mode
			
			# MODE 0: fix light and render multiview
			video_frames = []
			if render_mode == 0:
				assert opts.olat_idx is not None
				olat_idx = opts.olat_idx
				# Frames
				save_dir = os.path.join(opts.outdir, f'render_vd', f'{olat_idx:03d}', scan_name)
				os.makedirs(save_dir, exist_ok=True)
				# Save Mask
				save_mask_dir = os.path.join(opts.outdir, 'render_vd_mask', scan_name)
				os.makedirs(save_mask_dir, exist_ok=True)
				masks_mv_list = []
				# Save Video
				save_video_filename = os.path.join(opts.outdir, 'render_vd', f'{scan_name}_{olat_idx:03d}.mp4')
				video_out = imageio.get_writer(save_video_filename, mode='I', fps=30, codec='libx264', ffmpeg_params=['-crf', '0'])
				
				w_frames = opts.w_frames
				camera_lookat_point = torch.tensor([0, 0, 0], device=device)
				
				#  Get all the cameras to render
				render_cameras = []
				for frame_idx in tqdm(range(w_frames)):
					pitch_range = 0.30
					yaw_range = 0.30
					pitch = 3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames))
					yaw = 3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames))
					# cam2world_pose = LookAtPoseSampler.sample(
					# 	3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)),
					# 	3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames)),
					# 	camera_lookat_point, radius=2.7, device=device)
					
					c = get_pose_video(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					render_cameras.append(c)

				
				#  Generate masks if they did not exist
				if opts.gen_mask_mode == 2 and len(os.listdir(save_mask_dir)) != w_frames:
					for cam_idx, render_pose in enumerate(render_cameras):
						olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=render_pose, mix_triplane=mix_triplane, dirs=dirs[0:1])
						invert_img_novel_view = (face_pool(olat_dict_novel_view['image_recon']) * 0.5 + 0.5).clamp(0, 1)
						save_image = 255 * invert_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()
						# from image_utils.infer_mask import generate_rmbg_mask
						mask = generate_rmbg_mask(save_image)
						cv2.imwrite(f'{save_mask_dir}/{cam_idx:03d}.png', mask)
						masks_mv_list.append(generate_rmbg_mask(save_image))
				else:
					assert len(os.listdir(save_mask_dir)) == w_frames, f'Invalid mask mode or incorrect number of masks'
					for cam_idx, render_pose in enumerate(render_cameras):
						mask = cv2.imread(f'{save_mask_dir}/{cam_idx:03d}.png')
						masks_mv_list.append(mask)
				
				for frame_idx in tqdm(range(w_frames)):
					c = render_cameras[frame_idx]
					olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane, dirs=dirs[olat_idx:olat_idx+1])
					olat_img_novel_view = face_pool(olat_dict_novel_view['image'])
					olat_img_novel_view = (olat_img_novel_view * 0.5 + 0.5).clamp(0, 1)
					olat_img_novel_view = olat_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()
					
					if opts.gen_mask_mode == 2:
						olat_img_novel_view = (olat_img_novel_view * 255) * (masks_mv_list[frame_idx]/255)
						
						olat_emap = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{inset_envmap_fnames[olat_idx]:03d}.png'))
						olat_emap = cv2.flip(olat_emap, 1)
						inset_emap = cv2.resize(olat_emap, (UV_inset_res[1] * scale, UV_inset_res[0] * scale), interpolation=cv2.INTER_AREA)
						inset_emap = add_border(inset_emap, border_size=1)
						olat_img_novel_view[y_offset - 2 * bs:y_offset + UV_inset_res[0] * scale, x_offset - 2 * bs:x_offset + UV_inset_res[1] * scale] = inset_emap
						
					
					video_out.append_data((olat_img_novel_view).astype(np.uint8))
					cv2.imwrite(f'{save_dir}/{frame_idx:03d}.png', olat_img_novel_view[:, :, ::-1])
				
				video_out.close()
			
			# MODE 1: fix pose and render all the olats
			if render_mode == 1:
				#  Save Frames
				save_dir = os.path.join(opts.outdir, f'render_all_olat', scan_name)
				os.makedirs(save_dir, exist_ok=True)
				
				for olat_idx in range(len(dirs)):
					# Desired Pose
					# yaw = 0
					# pitch = 0
					# c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
			
					
					olat_dict = net.forward_eval(real_img, rec_ws=rec_ws, c=real_label, mix_triplane=mix_triplane, dirs=dirs[olat_idx:olat_idx+1])
					
					olat_img = face_pool(olat_dict['image'])
					olat_img = (olat_img * 0.5 + 0.5).clamp(0, 1)
					olat_img = olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy()
					
					#  TODO: add mask as input relighting
					olat_emap = cv2.imread(os.path.join(opts.envmap_zspiral_path, f'{inset_envmap_fnames[olat_idx]:03d}.png'))
					olat_emap = cv2.flip(olat_emap, 1)
					inset_emap = cv2.resize(olat_emap, (UV_inset_res[1] * scale, UV_inset_res[0] * scale), interpolation=cv2.INTER_AREA)
					inset_emap = add_border(inset_emap, border_size=1)
					np_img = (255*olat_img[:, :, ::-1]) * ref_m
					np_img[y_offset - 2 * bs:y_offset + UV_inset_res[0] * scale, x_offset - 2 * bs:x_offset + UV_inset_res[1] * scale] = inset_emap
					
					cv2.imwrite(f'{save_dir}/{olat_idx:03d}.png', np_img)
			
			if render_mode == 2:
				#  Save Frames
				save_dir = os.path.join(opts.outdir, f'render_rotation', scan_name)
				os.makedirs(save_dir, exist_ok=True)
				#  Save Mask
				save_mask_dir = os.path.join(opts.outdir, 'render_rotation_mask', scan_name)
				os.makedirs(save_mask_dir, exist_ok=True)
				masks_mv_list = []


				mp4 = (save_dir, scan_name)
				# w_frames = opts.w_frames
				w_frames = 180
				camera_lookat_point = torch.tensor([0, 0, 0], device=device)
				
				emap_config = sorted(glob.glob1(emap_path, '*.exr'))
				emap_list = emap_config[0:0+25]
				emap_ids = [emap_id for emap_id in emap_list]
				
				camera_lookat_point = torch.tensor([0, 0, 0], device=device)
				
				#  Get all the cameras to render
				render_cameras = []
				for frame_idx in tqdm(range(w_frames)):
					pitch_range = 0.25
					yaw_range = 0.25
					pitch = 3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames))
					yaw = 3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames))
					c = get_pose_video(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					render_cameras.append(c)
				
				#  Generate masks if they did not exist
				if opts.gen_mask_mode == 2 and len(os.listdir(save_mask_dir)) != w_frames:
					for cam_idx, render_pose in enumerate(render_cameras):
						olat_dict_novel_view = net.forward_eval(real_img, rec_ws=rec_ws, c=render_pose, mix_triplane=mix_triplane, dirs=dirs[0:1])
						invert_img_novel_view = (face_pool(olat_dict_novel_view['image_recon']) * 0.5 + 0.5).clamp(0, 1)
						save_image = 255 * invert_img_novel_view.squeeze().permute(1, 2, 0).data.cpu().numpy()
						# from image_utils.infer_mask import generate_rmbg_mask
						mask = generate_rmbg_mask(save_image)
						cv2.imwrite(f'{save_mask_dir}/{cam_idx:03d}.png', mask)
						masks_mv_list.append(generate_rmbg_mask(save_image))
				else:
					assert len(os.listdir(save_mask_dir)) == w_frames, f'Invalid mask mode or incorrect number of masks'
					for cam_idx, render_pose in enumerate(render_cameras):
						mask = cv2.imread(f'{save_mask_dir}/{cam_idx:03d}.png')
						masks_mv_list.append(mask)
				
				for frame_idx in tqdm(range(w_frames)):
					c = render_cameras[frame_idx]
					olats_imgs_arr = []
					for olat_idx in range(len(dirs)):
						olat_dict = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane,
						                             dirs=dirs[olat_idx:olat_idx + 1])
						olat_img = (olat_dict['image'] * 0.5 + 0.5).clamp(0, 1)
						# olat_img = (face_pool(rec_img_dict['image_raw']) * 0.5 + 0.5).clamp(0, 1)
						olats_imgs_arr.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
					
					olats_imgs_arr = np.array(olats_imgs_arr)
					
					if frame_idx == 0:
						im_min_arr = np.zeros(shape=[len(emap_ids)], dtype=np.float32)
						im_max_arr = np.zeros(shape=[len(emap_ids)], dtype=np.float32)
					
					for emap_idx, emap_name in enumerate(emap_list):
						
						save_path = os.path.join(save_dir, emap_name[:-4])
						os.makedirs(save_path, exist_ok=True)
						from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
						from_emap = (from_emap - from_emap.min()) / (from_emap.max() - from_emap.min())
						
						from_emap = np.resize(from_emap, (UV_res[0], UV_res[1], 3))
						emap = sample_uv_from_envmap(from_emap, emap_mask_array, fn=opts.emap_sample_fn)
						
						emap = emap[:, None, None, :]

						# Multiply OLATs with envmap
						relit_image = np.sum(emap * (olats_imgs_arr ** 2), axis=0)
						if frame_idx == 0:
							im_min_arr[emap_idx] = relit_image.min()
							im_max_arr[emap_idx] = relit_image.max()
						
						relit_image = (relit_image - im_min_arr[emap_idx]) / (im_max_arr[emap_idx] - im_min_arr[emap_idx])
						relit_image = np.sqrt(relit_image)
						relit_image_8bit = relit_image.clip(0, 1) * 255
						
						relit_image_8bit = relit_image_8bit * (masks_mv_list[frame_idx] / 255)
						
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
						# relit_image_8bit = eye_masks.mask_frame(frame_idx, relit_image_8bit)
						cv2.imwrite(os.path.join(save_path, f'{frame_idx:04d}.png'), relit_image_8bit[:, :, ::-1])
			
			# Mode 3: rotating env maps sequence within the inline code
			if render_mode == 3:
				mp4 = (save_dir, scan_name)
				w_frames = 1
				camera_lookat_point = torch.tensor([0, 0, 0], device=device)
				save_path = os.path.join(mp4[0], mp4[1], 'rotate-cam01')
				os.makedirs(save_path, exist_ok=True)
				
				emap_config = sorted(glob.glob1(emap_path, '*.exr'))
				emap_list = emap_config
				olat_img = None
				
				tracking_frames = np.empty(shape=[w_frames, 512, 512, 3], dtype=np.uint8)
				for frame_idx in tqdm(range(w_frames)):
					# c = real_label
					
					# pitch = 0.0
					# yaw = 0.0
					
					pitch = -0.1
					yaw = -0.28
					c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					
					# cam2world_pose = LookAtPoseSampler.sample(
					#     3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (w_frames)),
					#     3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (w_frames)),
					#     camera_lookat_point, radius=2.7, device=device)
					# # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
					# c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
					
					olat_dict = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane,
					                             dirs=dirs[0:0 + 1])
					
					recon_image = (olat_dict['image_recon'] * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255
					recon_image = recon_image.data.cpu().numpy().astype(np.uint8)[:, :, ::-1]
					tracking_frames[frame_idx] = recon_image
				
				print("Masks Generated of Shape : ", tracking_frames.shape)
				# Get all the eye mask
				from video_uitls import masking
				eye_masks = masking.GaussianEyeMasks(video=tracking_frames)
				
				
				for frame_idx in tqdm(range(w_frames)):
					imgs = []
					
					# Desired Pose
					# pitch = -0.15
					# yaw = -0.35
					# c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					
					# cam2world_pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 - 0.05 + pitch,
					#                                           camera_lookat_point, radius=2.7, device=device)
					#
					# # all_poses.append(cam2world_pose.squeeze().cpu().numpy())
					# # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
					# c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
					
					# interp = grid[yi][xi]
					# w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
					
					# loop over all the OLATs
					if olat_img is None:
						olats_imgs_arr = []
						for olat_idx in range(len(dirs)):
							olat_dict = net.forward_eval(real_img, rec_ws=rec_ws, c=c, mix_triplane=mix_triplane,
							                             dirs=dirs[olat_idx:olat_idx + 1])
							olat_img = (olat_dict['image'] * 0.5 + 0.5).clamp(0, 1)
							olats_imgs_arr.append(olat_img.squeeze().permute(1, 2, 0).data.cpu().numpy())
						
						olats_imgs_arr = np.array(olats_imgs_arr)
					
					# Modify enviroment maps
					roll_steps = 10
					for emap_name in emap_list[:20]:
						for roll_idx in range(roll_steps):
							from_emap_raw = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)[:, :, ::-1]
							from_emap_raw = np.roll(from_emap_raw, roll_idx, axis=1)
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
							emap_png = np.roll(emap_png, roll_idx, axis=1)
							image_20x40 = cv2.resize(emap_png, (20 * scale, 10 * scale), interpolation=cv2.INTER_AREA)
							
							# Determine the position to place the 20x40 image in the lower right corner
							x_offset = 512 - 20 * scale  # Calculate the x-offset for the lower right corner
							y_offset = 512 - 10 * scale  # Calculate the y-offset for the lower right corner
							
							# Copy the 20x40 image into the lower right corner of the 512x512 image
							relit_image_8bit[y_offset:y_offset + 10 * scale, x_offset:x_offset + 20 * scale] = image_20x40
							relit_image_8bit = eye_masks.mask_frame(frame_idx, relit_image_8bit)
							cv2.imwrite(os.path.join(save_path, f'{emap_name[:-4]}_{roll_idx:04d}.png'),
							            relit_image_8bit[:, :, ::-1])
					# relit_image = torch.tensor(relit_image_8bit.transpose(2, 0, 1), dtype=torch.float32) / 255 * 2 - 1
				#     imgs.append(relit_image)
				#
				#     video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
				# video_out.close()
		
		if opts.shape:
			mesh_path = os.path.join(save_dir, 'mesh.mrc')
			gen_mesh(G = net.decoder, ws = None, triplane=mix_triplane, save_path=mesh_path, device=device)
		
		if opts.edit:
			if "edit_d" not in locals():
				## EG3D W space attribution entanglement when applying InterFaceGAN
				if opts.edit_attr != "glass":
					edit_d = np.load(os.path.join("../example/ws_edit", opts.edit_attr+".npy"))
					edit_d_glass = np.load(os.path.join("../example/ws_edit", "glass.npy"))
					edit_d = opts.alpha*edit_d - 0.8*edit_d_glass
				else:
					edit_d = np.load(os.path.join("../example/ws_edit", opts.edit_attr+".npy"))
					edit_d = opts.alpha*edit_d
				
				edit_d = torch.tensor(edit_d).to(device)
			
			edit_ws = rec_ws + edit_d
			img_edit_dict, img_edit_dict_w = net.edit(rec_ws, edit_ws, real_img_512, real_label)
			mix_triplane_edit = img_edit_dict["mix_triplane"]
			edit_img = face_pool(img_edit_dict["image"])
			torchvision.utils.save_image(torch.cat([real_img_512, edit_img]), os.path.join(save_dir, f'edit_img_{opts.edit_attr}_{opts.alpha}.jpg'),
			                             padding=0, normalize=True, range=(-1,1))
			
			if opts.multi_view:
				imgs_multi_view = []
				coef = [1, 0, -1]
				for j in range(3):
					yaw = coef[j] * np.pi*25/360
					pitch = 0
					c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)
					
					img_dict_novel_view = net.decoder.synthesis(ws=edit_ws, c=c, triplane=mix_triplane_edit, forward_triplane=True, noise_mode='const')
					img_novel_view= face_pool(img_dict_novel_view["image"])
					imgs_multi_view.append(img_novel_view)
				
				torchvision.utils.save_image(torch.cat(imgs_multi_view), os.path.join(save_dir, f'multi_view_edit_{opts.edit_attr}_{opts.alpha}.jpg'),
				                             padding=0, normalize=True, range=(-1,1))
			
			
			if opts.video:
				video_path = os.path.join(save_dir, f'video_edit_{opts.edit_attr}_{opts.alpha}.mp4')
				camera_lookat_point = torch.tensor(net.decoder.rendering_kwargs['avg_camera_pivot'], device=device)
				gen_interp_video(net.decoder, edit_ws, triplane=mix_triplane_edit,
				                 mp4=video_path, image_mode='image', device=device, w_frames=opts.w_frames)



if __name__=="__main__":
	parser = get_parser()
	opts = parser.parse_args()
	
	print("="*50, "Using CUDA: " + opts.cuda, "="*50)
	os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda
	device = torch.device('cuda:' + opts.cuda)
	
	if os.path.isdir(opts.R_ckpt):
		ckpt_dir = opts.R_ckpt
	else:
		ckpt_dir = os.path.dirname(opts.R_ckpt)
	ckpts = natsorted(glob.glob1(ckpt_dir, '*.pkl'))
	if len(ckpts) == 0:
		raise FileNotFoundError(f'No .pkl checkpoints found in {ckpt_dir}')
	opts.R_ckpt = os.path.join(ckpt_dir, ckpts[-1])
	print(f'Loading {ckpts[-1]}')
	
	
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	infer_main(opts, device, now)
