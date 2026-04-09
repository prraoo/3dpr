import os
import glob
from natsort import natsorted
from PIL import Image, ImageOps

def resize_image(img):
	resized_img = img.resize((768, 768))
	# Calculate padding sizes
	left = top = (1024 - 768) // 2
	right = bottom = 1024 - 768 - left

	# Pad the resized image to 1024x1024
	padded_image = ImageOps.expand(resized_img, (left, top, right, bottom), fill='black')
	return padded_image

if __name__ == '__main__':
	data_mode = 1
	SAVE_DIR = f'/CT/VORF_GAN3/work/code/TotalRelighting/{data_mode:02d}/'
	os.makedirs(SAVE_DIR, exist_ok=True)

	if data_mode == 0:
		DATA_DIR = f'/CT/VORF_GAN3/work/code/TotalRelighting/LS_subjects/'
		image_fnames = natsorted(glob.glob1(DATA_DIR, '*.png'))
		for fname in image_fnames:
			scan_name, emap = fname.split('_')
			print(scan_name)
			original_image = Image.open(os.path.join(DATA_DIR, fname))
			print(original_image.size)
			final_image = resize_image(original_image)
			print(final_image.size)
			final_image.save(os.path.join(SAVE_DIR, f'{scan_name}.png'))
	else:
		DATA_DIR = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		ID_LIST = [615, 616, 617, 631, 654, 662, 707, 763, 765, 771, 800, 801, 808, 811, 814, 827]

		for scan_id in ID_LIST:
			fpath = os.path.join(DATA_DIR, f'ID00{scan_id:03d}', 'crop', f'Cam07_ID00{scan_id:03d}_EMAP-999.png')
			assert os.path.isfile(fpath), f'{scan_id} not found!'
			scan_name = f'ID00{scan_id:03d}'
			original_image = Image.open(fpath)
			print(original_image.size)
			final_image = resize_image(original_image)
			print(final_image.size)
			final_image.save(os.path.join(SAVE_DIR, f'{scan_name}.png'))



