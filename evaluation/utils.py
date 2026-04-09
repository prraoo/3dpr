"""
Image Preprocessing Helper Functions
"""
import numpy as np
import cv2

def resize_n_crop_numpy(img, scale_crop_param, target_size=1024, output_size=512, mask=None):
	# scale_crop_params = np.loadtxt(name)
	# img_name = os.path.basename(name).replace('.txt', '_EMAP-350.png')
	# print(img_name)
	# img_path = os.path.join(BASE_PATH, 'normal', img_name)
	# img = cv2.imread(img_path)
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

	# asd = from_OLAT[dy:dy + sh, dx:dx + sw].shape
	# qwe = img[sy:sy + sh, sx:sx + sw].shape

	from_OLAT[dy:dy + sh, dx:dx + sw] = img[sy:sy + sh, sx:sx + sw]

	# Center Crop
	center_crop_size = 700
	output_size = 512

	left = int(target_size / 2 - center_crop_size / 2)
	upper = int(target_size / 2 - center_crop_size / 2)
	right = left + center_crop_size
	lower = upper + center_crop_size

	return cv2.resize(from_OLAT[upper:lower, left:right], (output_size, output_size))
