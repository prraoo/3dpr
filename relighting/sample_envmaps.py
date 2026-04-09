import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import glob
from natsort import natsorted
import numpy as np



def get_lightdirs():
	light_dirs = np.loadtxt('/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/LSX_light_positions_mpi.txt').astype(np.float32)
	return light_dirs

def mpi2usc_mapping(full_lit_frames):
	usc_mapping = []
	count = 0
	for i in range(350):
		if i not in full_lit_frames:
			usc_mapping.append(count)
			count += 1
		else:
			usc_mapping.append(-1)
	return np.array(usc_mapping, dtype=np.int16)

if __name__ == '__main__':
	z_sprial_mask_paths = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/envmap_zspiral/'
	envmap_path = '/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0/'
	
	envmaps_list = natsorted(glob.glob(envmap_path + '*.exr'))
	z_sprial_masks_list = natsorted(glob.glob(z_sprial_mask_paths + '*.png'))
	
	full_light_indices = [0, 20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230, 251, 272, 293, 314, 335, 348, 349]
	
	mapping = mpi2usc_mapping(full_lit_frames=full_light_indices)
	
	print(mapping)
	print('\n\n')
	
	bad_light_indices = [270, 296, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]
	# bad_light_indices = []
	
	print(mapping[bad_light_indices])
	import pdb; pdb.set_trace()
	dirs = get_lightdirs()
	
	all_light_indices = [x for x in range(len(dirs)) if (x not in full_light_indices and x < 350)]
	all_light_indices = [x for x in all_light_indices if x not in bad_light_indices]
	
	
	output_dir = '/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/training-MPI-LS/04_runs/envmap_sampling/'
	
	for idx, envmap_fname in enumerate(envmaps_list):
		if idx >= 1:
			break
		envmap = cv2.imread(envmap_fname, -1)
		envmap = cv2.resize(envmap, (512, 256))
		save_emap_dir = os.path.join(output_dir, os.path.basename(envmap_fname).split('.')[0])
		os.makedirs(save_emap_dir, exist_ok=True)
		
		for uv_idx, z_sprial_mask in enumerate(z_sprial_masks_list):
			mask = cv2.imread(z_sprial_mask)
			sample_emap = envmap * mask
			max_1 = sample_emap.mean(axis=(0,1))
			sample_emap = cv2.resize(sample_emap, (20, 10))
			max_2 = sample_emap.mean(axis=(0,1))
			print(max_1, max_2, sample_emap.shape)
			
			cv2.imwrite(os.path.join(save_emap_dir, f'{uv_idx:03d}.png'), sample_emap)
		
		
		
