import cv2
import numpy as np
import os
import glob
from natsort import natsorted

tonemap_olat = lambda x: pow(x, 0.5)


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


landmarks_path = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
olat_path = '/HPS/FacialRelighting/nobackup/data/FOLAT_c2/'
scan_name = 'ID00088'
cam_names = [f'Cam{x:02d}' for x in range(16)]
print(cam_names)

for cam_name in cam_names:
	output_path = os.path.join('/CT/VORF_GAN3/work/code/VoRF/gt_olats/', scan_name)
	os.makedirs(output_path, exist_ok=True)

	full_olat_path = os.path.join(olat_path, cam_name, scan_name)
	olat_imgs = natsorted(glob.glob1(full_olat_path, '*.png'))

	assert len(olat_imgs) == 150
	# scale_crop_params = np.loadtxt(os.path.join(landmarks_path, scan_name, 'transform', f'{cam_name}_{scan_name}.txt'))

	for idx, fname in enumerate(olat_imgs):
		if idx != 23:
			continue
		img = cv2.imread(os.path.join(full_olat_path, fname), -1)

		# img = resize_n_crop_numpy(img, scale_crop_params)
		img = tonemap_olat(img)

	cv2.imwrite(os.path.join(output_path, f'{cam_name}_{idx:03d}.png'), img)

