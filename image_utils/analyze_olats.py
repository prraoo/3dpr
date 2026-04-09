import os
import glob
import numpy as np
from natsort import natsorted
import cv2
import argparse
from distutils.util import strtobool
import OpenEXR
import Imath
import sys
sys.path.insert(0, '.')
from image_utils.data_util import load_exr, save_exr

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"



def apply_tonemap(hdr_image, operator='reinhard', key=0.18, gamma=2.2, white=1.0, scale_luminance=True):
	"""
	Applies tone mapping to an HDR image using the specified operator.

	Parameters:
		hdr_image (numpy.ndarray): The input HDR image.
		operator (str): The tone mapping operator to use ('reinhard' or 'extended_reinhard').
		key (float): Key value for the extended Reinhard operator (default is 0.18).
		gamma (float): Gamma correction value (default is 2.2).
		gamma (float): Gamma correction value (default is 2.2).
		white (float): White point value (default is 1.0).
		scale_luminance (bool): Option to logarithmic average of luminance (default is False).

	Returns:
		numpy.ndarray: The tone-mapped image.
	"""

	# Compute luminance from RGB values using Rec. 709 coefficients
	luminance = 0.2126 * hdr_image[:, :, 2] + \
				0.7152 * hdr_image[:, :, 1] + \
				0.0722 * hdr_image[:, :, 0]

	# Avoid division by zero
	epsilon = 1e-8
	
	# Compute the logarithmic average luminance
	delta = epsilon
	if scale_luminance:
		log_average_luminance = np.exp(np.mean(np.log(luminance + delta)))
		print(log_average_luminance)
		L_scaled = (key / log_average_luminance) * luminance
	else:
		L_scaled = luminance
		
	
	if operator == 'rhd':
		# Basic Reinhard operator
		# L_mapped = L / (1 + L)
		L_mapped = L_scaled / (1.0 + L_scaled)

	elif operator == 'ext_rhd':
		# Extended Reinhard operator
		
		# Scale luminance using the key value
		L_mapped = (L_scaled * (1 + (L_scaled / (white ** 2)))) / (1.0 + L_scaled)
		
	else:
		raise ValueError("Unsupported operator. Choose 'reinhard' or 'extended_reinhard'.")

	# Compute the ratio of mapped luminance to original luminance
	luminance_ratio = (L_mapped + epsilon) / (luminance + epsilon)

	# Apply the ratio to each RGB channel
	tonemapped_image = hdr_image.copy() * luminance_ratio[:, :, np.newaxis]

	# Apply gamma correction
	tonemapped_image = np.clip(tonemapped_image, 0, 1)
	tonemapped_image = np.power(tonemapped_image, 1.0 / gamma)

	return tonemapped_image


def convert_to_16bit(tonemapped_image, save_fname):
	# Scale to 16-bit integer range [0, 65535]
	image_16bit = np.uint16(tonemapped_image * 65535)
	cv2.imwrite(save_fname, image_16bit[:, :, ::-1])


def render_olats(image_fname, mask, save_dir, dataset, tmo):
	if dataset == 'mpi':
		image = load_exr(image_fname)
		image = image.astype(np.float32)
		image = np.clip(image, 0, 5)
		# original = image.copy()
		
		# p2 = apply_tonemap(image, operator=tmo, key=0.001, white=0.12, scale_luminance=False)
		# from image_utils.tonemap import apply_tonemap
		p2 = apply_tonemap(image, operator=tmo, key=0.001, white=0.12, scale_luminance=False)
		
		final_image_exr = cv2.hconcat([p2], 1)
		
		fname = os.path.basename(image_fname)
		# save_exr(os.path.join(save_dir, 'exr', fname), final_image_exr)
		
		convert_to_16bit(final_image_exr, os.path.join(save_dir, 'png', fname.replace('.exr', '.png')))
		
		# print(f" {os.path.basename(image_fname)} {original.min():0.06f}, {original.max():0.06f}, {p4.min():0.06f}, {p4.max():0.06f}")
		# print(f" {os.path.basename(image_fname)} - Scale = {scale:0.06f}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_name", type=str, default='mpi')
	parser.add_argument("--tmo", type=str, default='rhd')
	
	args = parser.parse_args()
	
	
	merl_olat_dataset = '/HPS/FacialRelighting/nobackup/data/FOLAT_c2/Cam06/ID00004/'
	mpi_olat_dataset  = '/CT/VORF_GAN4/static00/datasets/FOLAT_c2_align/Cam06/ID20001/'
	mask = cv2.imread('/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/ID20001/sam/Cam07_ID20001.png')
	mask = mask/255.
	
	
	save_dir = '/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/image_utils/debug/'
	os.makedirs(save_dir, exist_ok=True)
	
	if args.dataset_name == 'merl':
		olat_images = natsorted(glob.glob(merl_olat_dataset + '*.png'))
	elif args.dataset_name == 'mpi':
		full_light_indices = [0, 20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230, 251, 272, 293, 314, 335, 348, 349]
		olat_images = natsorted(glob.glob(mpi_olat_dataset + '*.exr'))
		olat_images = [fname for fname in olat_images if int(os.path.splitext(os.path.basename(fname))[0]) not in full_light_indices]
	else:
		raise ValueError('Dataset not supported')
	
	
	for olat_fname in olat_images:
		# Save image
		render_olats(olat_fname, mask, save_dir, args.dataset_name, args.tmo)

		
		