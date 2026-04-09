import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import concurrent.futures
import numpy as np
import cv2
import sys
sys.path.append('.')
from image_utils.tonemap import apply_tonemap as process_image
from image_utils.data_util import resize_n_crop_numpy
import glob
from natsort import natsorted


def read_numbers_from_file(file_path):
	numbers = []
	try:
		with open(file_path, 'r') as file:
			for line in file:
				# Convert each line to a number, assuming they are integers
				# Use float(line.strip()) if the numbers are floating-point
				numbers.append(int(line.strip()))
	except Exception as e:
		print(f"An error occurred: {e}")
	return numbers


def load_HDR_image(path, params):
	assert os.path.exists(path), f"{path} does not exist"
	image = cv2.imread(path, -1)
	image = process_image(image)
	image = (image / 65535)
	if params is not None:
		image = resize_n_crop_numpy(image, params)
	return image


def load_image_parallel(image_path, light_indices, cam_name='Cam06', subject_id='ID20001', params=None, ls_res=2, dirs=None, n_workers=16):
	data = []
	olat_fnames = [os.path.join(image_path, cam_name, subject_id, f'{idx:03d}.exr' ) for idx in light_indices]
	olat_fnames = [f for i, f in enumerate(olat_fnames) if i % ls_res == 0]
	
	# for idx, f in enumerate(olat_fnames):
	# 	print(f'{idx}: {os.path.basename(f)}: {dirs[idx].data.cpu()}')
	#
	data = []
	for image_path in olat_fnames:
		image = load_HDR_image(image_path, params)
		data.append(image)
	
	# import pdb; pdb.set_trace()
	# with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:  # Adjust number of workers as needed
	# 	# Submit tasks to the executor for each OLAT_id
	# 	future_to_olat = {executor.submit(load_HDR_image, image_path, params): image_path for image_path in olat_fnames}
	#
	# 	# Collect results
	# 	for future in concurrent.futures.as_completed(future_to_olat):
	# 		try:
	# 			result = future.result()
	# 			data.append(result)  # Append the result to the shared data list
	# 		except Exception as exc:
	# 			print(f"OLAT {future_to_olat[future]} generated an exception: {exc}")
	assert len(data) == len(olat_fnames)
	return np.array(data)


def load_envmap(env_map, mapy,order, light_idx):
	mask = np.zeros((256, 512, 3)).astype('uint8')
	mask[mapy == (order[light_idx] - 1)] = 1
	masked = env_map * mask
	intensity_sum = np.sum(masked, axis=2)
	max_intensity_location = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)
	color = masked[max_intensity_location]
	return color


def load_envmap_parallel(env_map_path, map, order, light_indices, n_workers=16):
	sample_indices = [idx for idx, val in enumerate(light_indices)]
	env_map = load_HDR_image(env_map_path, params=None)
	mapy = np.asarray(read_numbers_from_file(map)).reshape(256, 512)
	order = np.asarray(read_numbers_from_file(order))
	
	emap = []
	for light_idx in sample_indices:
		result = load_envmap(env_map, mapy, order, light_idx)
		emap.append(result)
	return np.array(emap)


def get_envmap_mask(zspiral_indices, n_olats, path):
	mask_array = []
	for olat_idx in range(n_olats):
		mask = cv2.imread(os.path.join(path, f'{zspiral_indices[olat_idx]:03d}.png')) / 255
		mask = cv2.flip(mask, 1)
		mask_array.append(mask)
	return np.array(mask_array)


def get_lightdirs(device, dataset_name):
	if dataset_name == 'mpi':
		light_dirs = np.loadtxt('/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/LSX_light_positions_mpi.txt').astype(np.float32)
	else:
		light_dirs = np.load('/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy').astype(np.float32)
	return light_dirs / np.linalg.norm(light_dirs, axis=1)[..., None]


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

if __name__ == '__main__':
	
	## Load All Paths
	emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/quant-eval/'
	# emap_path = '/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0_sparse_random-all/'
	map_file = '/CT/LS_FRM01/work/studio-tools/LightStage/relighting/LSX_data/Light_Probe_Mapping_MPI.txt'
	lights_order_file = '/CT/LS_FRM01/work/studio-tools/LightStage/relighting/LSX_data/LSX3_light_z_spiral.txt'
	input_dir ='/CT/VORF_GAN5/static00/datasets/FOLAT_c2_align/'
	use_inset = False
	# sub_id = sys.argv[1]
	sub_id = 'ID20010'
	# output_dir = f'/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test-no_inset/'
	# output_dir = f'/CT/VORF_GAN5/static00/datasets/evaluation-sparse_envmaps/'
	output_dir = f'/CT/VORF_GAN5/nobackup/datasets/evaluation/CVPR26-debug_03/'
	envmap_zspiral_path = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/envmap_zspiral_mpi/'
	lightstage_res = 'half'
	emap_sample_fn = 'mean'
	use_crop_align = True

	# sub_id = 'ID20010'
	# cam = 'Cam06'
	cam_list = [4, 6, 7, 23, 39]
	# cam_list = [6]
	
	for view_idx, cam_id in enumerate(cam_list):
		cam = f'Cam{cam_id:02d}'
	
		lm_file = f'/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/{sub_id}/transform/{cam}_{sub_id}.txt'
		mask_file = f'/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/{sub_id}/bgMatting/{cam}_{sub_id}.png'
		
		assert os.path.exists(lm_file), f'The {cam} view for {sub_id} was not detected!'
		
		# Masking and cropping params
		scale_crop_params = np.loadtxt(lm_file)
		ref_m = cv2.imread(mask_file) / 255
		ref_m_cropped = resize_n_crop_numpy(ref_m,scale_crop_params)

		dirs = get_lightdirs(device='cuda', dataset_name='mpi')
		ls_freq = 1 if lightstage_res == 'full' else 2
		
		# Load lightstage configurations
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
		n_olats = len(dirs)
		
		# Select needed inset envmaps
		inset_envmap_fnames = inset_envmap_fnames[all_light_indices]
		inset_envmap_fnames = inset_envmap_fnames[select_indices]
		assert len(inset_envmap_fnames) == n_olats, 'Envmaps do not match lights'
		
		# Load OLATs
		olat_imgs_arr = load_image_parallel(input_dir, all_light_indices, subject_id=sub_id, cam_name=cam, params=None, ls_res=ls_freq)
		# _, h, w, _ = olat_array.shape
		print(olat_imgs_arr.shape, olat_imgs_arr[2].max())
		
		# Load Envmap Masks
		emap_mask_array = get_envmap_mask(inset_envmap_fnames, n_olats, envmap_zspiral_path)
		
		emap_config = natsorted(glob.glob1(emap_path, '*.exr'), reverse=False)
		emap_list = emap_config[:95]
		
		save_image_dir = os.path.join(output_dir, f'{sub_id}', 'images')
		os.makedirs(save_image_dir, exist_ok=True)
		if use_crop_align == True:
			save_image_dir_crop = os.path.join(output_dir, f'{sub_id}', 'images-crop')
			os.makedirs(save_image_dir_crop, exist_ok=True)
		
		UV_res = [13, 26]  # 14*28=392
		UV_inset_res = [13, 26]  # a white background
		scale = 8
		x_offset = 512 - UV_inset_res[1] * scale  # Calculate the x-offset for the lower right corner
		y_offset = 512 - UV_inset_res[0] * scale  # Calculate the y-offset for the lower right corner
		
		for emap_idx, emap_name in enumerate(emap_list):
			from_emap = cv2.imread(os.path.join(emap_path, emap_name), -1).astype(np.float32)
			from_emap = (from_emap - from_emap.min()) / (from_emap.max() - from_emap.min())
			# from_emap = np.ones_like(from_emap)
			
			# Relighting with sampled envmap masks
			from_emap = np.resize(from_emap, (UV_res[0], UV_res[1], 3))
			emap = sample_uv_from_envmap(from_emap, emap_mask_array,fn=emap_sample_fn)
			
			if not np.isnan(emap.any()):
				# Multiply OLATs with envmap
				# relit_image = np.mean(emap * (olat_imgs_arr**2), axis=0)
				relit_image = np.sum(emap[:, None, None, :] * (olat_imgs_arr ** 2), axis=0)
				if view_idx == 0:
					im_max = relit_image.max()
					im_min = relit_image.min()
				relit_image = (relit_image - im_min) / (im_max - im_min)
				relit_image = np.sqrt(relit_image)
				relit_image_8bit = relit_image.clip(0, 1) * 255
				
				np_img = (relit_image_8bit).astype(np.uint8)
				cv2.imwrite(f'{save_image_dir}/{cam}_{sub_id}_{emap_name[:-4]}.png', np_img)
				
				if use_crop_align == True:
					np_img = resize_n_crop_numpy(np_img, scale_crop_params)
					if use_inset:
						# Create the inset envmap image to be added
						emap_png = cv2.imread(os.path.join(emap_path, emap_name.replace('exr', 'png')))
						emap_inset = cv2.resize(emap_png, (UV_inset_res[1] * scale, UV_inset_res[0] * scale), interpolation=cv2.INTER_AREA)
						
						np_img[y_offset:y_offset + UV_inset_res[0] * scale,
						x_offset:x_offset + UV_inset_res[1] * scale] = emap_inset
					
					cv2.imwrite(f'{save_image_dir_crop}/{cam}_{sub_id}_{emap_name[:-4]}.png', np_img)

			
