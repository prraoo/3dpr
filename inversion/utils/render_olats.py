import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import cv2
import glob
from natsort import natsorted

#TODO: Crop and align
def resize_n_crop_numpy(img, scale_crop_param, target_size=1024, output_size=512, mask=None):
	C = img.shape[-1]
	t = np.array([-1, -1])
	w0, h0, s, t[0], t[1] = scale_crop_param
	target_size = 1024
	# Scale
	w = (w0 * s).astype(np.int16)
	h = (h0 * s).astype(np.int16)
	# Crop
	x = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int16)
	right = x + target_size
	y = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int16)
	below = y + target_size

	img = cv2.resize(img, (w, h))
	from_OLAT = np.zeros(shape=(target_size, target_size, C), dtype=img.dtype)

	sx = x if x >= 0 else 0
	sy = y if y >= 0 else 0
	dx = 0 if x >= 0 else -x
	dy = 0 if y >= 0 else -y

	sw = min(right - sx, w - sx)
	sh = min(below - sy, h - sy)
	from_OLAT[dy:dy + sh, dx:dx + sw] = img[sy:sy + sh, sx:sx + sw]

	# Center Crop
	center_crop_size = 700
	output_size = 512

	left = int(target_size / 2 - center_crop_size / 2)
	upper = int(target_size / 2 - center_crop_size / 2)
	right = left + center_crop_size
	lower = upper + center_crop_size

	return cv2.resize(from_OLAT[upper:lower, left:right], (output_size, output_size))

#TODO: uv sampling

def get_dir2uv(light_dirs):
	# # TODO: revise the calculations
	# _theta = np.arctan2(light_dirs[:, 1], light_dirs[:, 0]) * 180 / np.pi
	# phi = np.arccos(light_dirs[:, 2] / np.linalg.norm(light_dirs, axis=1)) * 180 / np.pi
	#
	# # get uv indices
	# neg_theta_idx = (_theta < 0).astype(int) * 360
	# theta = (_theta + neg_theta_idx) / 18
	# theta = theta.astype(int)
	#
	# phi = phi / 18
	# phi = phi.astype(int)
	#
	# # 1D indices
	# indices = (20 * phi) + theta
	# return indices.astype(int)

	import math
	uv = []
	U = 10
	V = 20
	for l in range(150):
		light = light_dirs[l]
		light = light.copy() / np.linalg.norm(light)

		u1 = 0.5 + math.atan2(light[0], light[2]) / (math.pi * 2)
		v1 = 1 - (0.5 + light[1] * 0.5)
		u = int(v1 * U)
		v = int(u1 * V)
		uv.append([u, v])

	return np.array(uv)

#TODO: Load all the OLATs
def load_olats(olat_path, scan_name, cam_name, landmarks_path):
	img_list = []
	scale_crop_params = np.loadtxt(os.path.join(landmarks_path,scan_name, 'transform', f'{cam_name}_{scan_name}.txt'))
	for fname in natsorted(glob.glob1(os.path.join(olat_path, cam_name, scan_name),'*.png')):
		img = cv2.imread(os.path.join(olat_path, cam_name, scan_name, fname), -1)
		img = resize_n_crop_numpy(img, scale_crop_params)
		img = img / 65535.
		img_list.append(img)


	assert len(img_list) == 150

	return np.array(img_list)


##TODO: Save

if __name__ == '__main__':
	OLAT_PATH = '/HPS/FacialRelighting/nobackup/data/FOLAT_c2/'
	LANDMARKS_PATH = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
	SCAN_NAME = f'ID00{sys.argv[1]}'
	CAM_NAME = 'Cam07'
	save_emap = False
	# SAVE_DIR = os.path.join('/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/reference/TotalRelighting/', CAM_NAME, SCAN_NAME)
	# SAVE_DIR = os.path.join('/CT/VORF_GAN3/work/code/TotalRelighting/00_gt')
	SAVE_DIR = os.path.join('/CT/VORF_GAN3/work/code/VoRF/relit_gt')
	os.makedirs(SAVE_DIR, exist_ok=True)
	if save_emap:
		SAVE_PHOTOAPP_ENV_DIR = os.path.join('/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/reference/relit/', 'envs')
		os.makedirs(SAVE_PHOTOAPP_ENV_DIR, exist_ok=True)

	light_dirs = np.load('/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy')
	light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1)[..., None]
	uv_idx = get_dir2uv(light_dirs)


	olat_array = load_olats(OLAT_PATH, SCAN_NAME, CAM_NAME, LANDMARKS_PATH)

	## Get Environment Maps
	# emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor_2018-ds/'
	emap_path = '/CT/VORF_GAN3/work/code/TotalRelighting/envs/'
	emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/indoor-ds-rot-new/'
	emap_list = natsorted(glob.glob1(emap_path, '*.exr'), reverse=False)
	emap_list = emap_list[:30]

	for emap_name in emap_list:
		from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)
		from_emap = np.resize(from_emap, (10, 20, 3))

		# # Relighting with incorrect uv sampling
		# env_map = from_emap.reshape(-1, 3)
		# env_map = env_map / env_map.max(0)
		# env_map = env_map[uv_idx]
		# print(env_map.shape)
		#
		# np.save(os.path.join(SAVE_PHOTOAPP_ENV_DIR, f'{emap_name}'.replace('exr', 'npy')), env_map)


		# Relighting with fixed lighting sampling
		from_emap = from_emap[None].repeat(150, axis=0)
		u_indices = uv_idx[:, 0].astype(int)  # First column contains u indices
		v_indices = uv_idx[:, 1].astype(int)  # Second column contains v indices
		from_emap = from_emap[np.arange(150), u_indices, v_indices]
		emap = from_emap[:, None, None, :]

		if not np.isnan(emap.any()):
			# Multiply OLATs with envmap
			relit_image = np.sum(emap * (olat_array), axis=0)
			im_max = relit_image.max()
			im_min = relit_image.min()
			relit_image = (relit_image - relit_image.min()) / (relit_image.max() - relit_image.min())
			relit_image = np.sqrt(relit_image)
			relit_image_8bit = relit_image.clip(0, 1) * 255

			# Create the 20x40 image to be added
			scale = 8
			emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))
			image_20x40 = cv2.resize(emap_png, (20 * scale, 10 * scale), interpolation=cv2.INTER_AREA)

			# Determine the position to place the 20x40 image in the lower right corner
			x_offset = 512 - 20 * scale  # Calculate the x-offset for the lower right corner
			y_offset = 512 - 10 * scale  # Calculate the y-offset for the lower right corner

			# Copy the 20x40 image into the lower right corner of the 512x512 image
			relit_image_8bit[y_offset:y_offset + 10 * scale, x_offset:x_offset + 20 * scale] = image_20x40
			cv2.imwrite(f'{SAVE_DIR}/{CAM_NAME}_{SCAN_NAME}_{emap_name[:-4]}.png',relit_image_8bit)


