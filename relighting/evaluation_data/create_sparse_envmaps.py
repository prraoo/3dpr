import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import cv2
import glob
from natsort import natsorted

if __name__ == '__main__':
	# emap_path        = '/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0/'
	emap_path        = '/HPS/prao2/static00/datasets/Environment-Maps/outdoor-ds-new/'
	# output_emap_path_row = '/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0_sparse_90_row/'
	# os.makedirs(output_emap_path_row, exist_ok=True)
	# output_emap_path_col = '/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0_sparse_90_col/'
	# os.makedirs(output_emap_path_col, exist_ok=True)
	sparsity = 10
	# output_emap_path_random = f'/HPS/prao2/static00/datasets/Environment-Maps/sigg-24-ppt-0_sparse_{sparsity}_random/'
	output_emap_path_random = f'/HPS/prao2/static00/datasets/Environment-Maps/outdoor-ds-new_{sparsity}_random/'
	os.makedirs(output_emap_path_random, exist_ok=True)

	emap_fnames_list = natsorted(glob.glob(emap_path + '*.exr'))
	
	# # Row_wise
	# emap = cv2.imread(emap_fnames_list[0], -1)
	#
	# emap_mask_01 = np.zeros_like(emap)
	# emap_mask_01[0:1, 10:12] = 1
	# emap_mask_02 = np.zeros_like(emap)
	# emap_mask_02[4:5, 10:12] = 1
	# emap_mask_03 = np.zeros_like(emap)
	# emap_mask_03[9:10, 10:12] = 1
	# row_masks = [emap_mask_01, emap_mask_02, emap_mask_03]
	#
	# for idx, row_mask in enumerate(row_masks):
	# 	for fname in emap_fnames_list:
	# 		emap = cv2.imread(fname, -1)
	# 		emap = cv2.resize(emap, (20, 10))
	# 		emap *= row_mask
	# 		emap_name = os.path.basename(fname)
	# 		cv2.imwrite(os.path.join(output_emap_path_row, f'{emap_name[:-4]}_row_{idx:02d}.exr'), emap)
	#
	# # Column Wise
	#
	# emap = cv2.imread(emap_fnames_list[0], -1)
	# emap_mask_11 = np.zeros_like(emap)
	# emap_mask_11[0:3, 0:2] = 1
	# emap_mask_12 = np.zeros_like(emap)
	# emap_mask_12[0:3, 5:7] = 1
	# emap_mask_13 = np.zeros_like(emap)
	# emap_mask_13[0:3, 9:11] = 1
	# emap_mask_14 = np.zeros_like(emap)
	# emap_mask_14[0:3, 14:16] = 1
	# emap_mask_15 = np.zeros_like(emap)
	# emap_mask_15[0:3, 18:] = 1
	# col_masks = [emap_mask_11, emap_mask_12, emap_mask_13, emap_mask_14, emap_mask_15]
	#
	# for idx, col_mask in enumerate(col_masks):
	# 	for fname in emap_fnames_list:
	#
	# 		emap = cv2.imread(fname, -1)
	# 		emap = cv2.resize(emap, (20, 10))
	# 		emap *= col_mask
	# 		emap_name = os.path.basename(fname)
	# 		cv2.imwrite(os.path.join(output_emap_path_col, f'{emap_name[:-4]}_col_{idx:02d}.exr'), emap)
	#
	#
	# randomly pick 3 pixels
	# Parameters
	height, width = 10, 20
	n = 3  # Number of pixels to set to 1
	
	
	# Flatten the image index range: total indices = 20 * 10
	total_pixels = height * width
	n = int(total_pixels * sparsity/100)
	
	n_envmaps = 10
	# n=5
	
	
	for idx in range(n_envmaps):
		# Create a 20x10 image of all zeros
		mask = np.zeros((height, width), dtype=np.uint8)
		# Randomly choose n unique indices from 0 to total_pixels - 1
		sampled_indices = np.random.choice(total_pixels, size=n, replace=False)
		
		# Set the sampled indices to 1 (convert flat indices to 2D indices)
		mask.flat[sampled_indices] = 1
		
		for fname in emap_fnames_list:
			emap = cv2.imread(fname, -1)
			emap = cv2.resize(emap, (20, 10))
			emap *= mask[:, :, np.newaxis]
			emap_name = os.path.basename(fname)
			cv2.imwrite(os.path.join(output_emap_path_random, f'{emap_name[:-4]}_random_{idx:02d}.exr'), emap)


