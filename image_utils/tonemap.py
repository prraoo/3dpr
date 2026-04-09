import os
import glob
import numpy as np
from natsort import natsorted
import cv2
import argparse
from distutils.util import strtobool
# import OpenEXR
# import Imath
import sys
sys.path.insert(0, '.')

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def apply_tonemap(hdr_image, operator='rhd', to_png=True, key=0.001, gamma=2.2, white=0.12, scale_luminance=False):
	"""
	Applies tone mapping to an HDR image using the specified operator.

	Parameters:
		hdr_image (numpy.ndarray): The input HDR image.
		operator (str): The tone mapping operator to use ('reinhard' or 'extended_reinhard').
		to_png (bool): convert to 16 but (default is False).
		key (float): Key value for the extended Reinhard operator (default is 0.18).
		gamma (float): Gamma correction value (default is 2.2).
		gamma (float): Gamma correction value (default is 2.2).
		white (float): White point value (default is 1.0).
		scale_luminance (bool): Option to logarithmic average of luminance (default is False).

	Returns:
		numpy.ndarray: The tone-mapped image.
	"""
	hdr_image = hdr_image.astype(np.float32)
	hdr_image = np.clip(hdr_image, 0, 5)
	
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

	if to_png:
		tonemapped_image = np.uint16(tonemapped_image * 65535)
	return tonemapped_image



