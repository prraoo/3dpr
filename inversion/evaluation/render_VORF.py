import os
import glob
from natsort import natsorted
import numpy as np
import cv2

# tonemap = lambda x :(pow(x/(pow(2,16)),0.4) * 255)
tonemap = lambda x :(pow(x/(pow(2,16)),0.5) * 256)

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


if __name__ == '__main__':
	BASE_PATH = '/CT/VORF_GAN3/work/code/VoRF/'
	CAM_LIST = ['Cam07', 'Cam06']
	SCAN_NAME = 'ID00651'

	save_path = os.path.join(os.path.join(BASE_PATH,'olats', SCAN_NAME))
	# save_path = os.path.join(os.path.join(BASE_PATH,'olats-normalized', SCAN_NAME))
	os.makedirs(save_path, exist_ok=True)

	for cam_name in CAM_LIST:
		olat_path = os.path.join(BASE_PATH, 'render-emaps-wild', cam_name, SCAN_NAME)
		olat_list = natsorted(glob.glob1(olat_path, '*.npy'))
		print(len(olat_list))

		# landmarks_path = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		# scale_crop_params = np.loadtxt(os.path.join(landmarks_path, SCAN_NAME, 'transform', f'{cam_name}_{SCAN_NAME}.txt'))

		for fname in olat_list:
			olat = np.load(os.path.join(olat_path, fname))
			# olat = (olat/ olat.max()).clip(0, 1)
			# ROTATE and RESIZE to 512x512
			olat = cv2.resize(olat, (1300, 1030))
			olat = cv2.rotate(olat, cv2.ROTATE_90_CLOCKWISE)
			# olat = resize_n_crop_numpy(olat, scale_crop_params)
			olat = tonemap(olat)
			# olat = (olat/ olat.max()).clip(0, 1) *255


			cv2.imwrite(os.path.join(save_path, f"{cam_name}_{SCAN_NAME}_{fname.replace('.npy', '.png')}"), olat[:, :, ::-1])

