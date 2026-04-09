import os
import numpy as np
import cv2
import glob
from natsort import natsorted


if __name__ == '__main__':
	z_spiral_envmap_path = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/envmap_zspiral'
	light_dirs_path = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/LSX_light_positions_mpi.txt'
	output_path = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/envmap_zspiral_mpi'
	os.makedirs(output_path, exist_ok=True)
	
	n_olats_mpi = 350  # 331 OLATs and rest tracking frames
	dirs = np.loadtxt(light_dirs_path)
	olat_count = 0
	
	for i in range(n_olats_mpi):
		light_dir = dirs[i]
		
		if light_dir[0] == 0 and light_dir[1] == 0 and light_dir[2] == 0:
			print(f"Tracking Frame {i}")
			# Save Tracking frame as zero image
			envmap = cv2.imread(os.path.join(z_spiral_envmap_path, f'L_{0:03d}.png'))
			cv2.imwrite(os.path.join(output_path, f'{i:03d}.png'), envmap*0)

		else:
			envmap = cv2.imread(os.path.join(z_spiral_envmap_path, f'L_{olat_count:03d}.png'))
			olat_count += 1
			cv2.imwrite(os.path.join(output_path, f'{i:03d}.png'), envmap)
	
	print(f'Total OLAT Count: {olat_count}')