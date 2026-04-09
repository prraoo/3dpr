import os
import sys
import glob

import numpy as np
from natsort import natsorted
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
	transforms.Resize(image_size),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#
# base_path = sys.argv[1]
# subject_id = sys.argv[2]
# input_path = os.path.join(base_path, subject_id, 'crop')
# mask_path = os.path.join(base_path, subject_id, 'mask_rmbg2')
#
# images_path = natsorted(glob.glob1(input_path, f'*'))
# os.makedirs(mask_path, exist_ok=True)
#
# for fname in images_path:
# 	image = Image.open(os.path.join(input_path, fname))
#
# 	input_images = transform_image(image).unsqueeze(0).to('cuda')
# 	# Prediction
# 	with torch.no_grad():
# 		preds = model(input_images)[-1].sigmoid().cpu()
# 		pred = preds[0].squeeze()
# 		pred_pil = transforms.ToPILImage()(pred)
# 		mask = pred_pil.resize(image.size)
# 		mask.save(os.path.join(mask_path, fname))

def generate_rmbg_mask(im):
	if type(im) is str:
		im = Image.open(im)
	elif type(im) is np.ndarray:
		im = Image.fromarray(im.astype(np.uint8))
	
	im_in = transform_image(im).unsqueeze(0).to('cuda')
	
	with torch.no_grad():
		preds = model(im_in)[-1].sigmoid().cpu()
		im_pred = preds[0].squeeze()
		im_pred_pil = transforms.ToPILImage()(im_pred)
		m = im_pred_pil.resize(im.size)
	return np.array(m)[..., None]


if __name__ == '__main__':
	base_path = '/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/training-MPI-LS/02_runs/00030-gpus4-batch14-87ids-32o_dim--1_lpips0.02-L1_loss-lcol2.0-lr0.0003/render-relight/invert_mv/ID00600'
	images_path = natsorted(glob.glob1(base_path, '*.jpg'))
	print(images_path)
	
	for fname in images_path:
		image = Image.open(os.path.join(base_path, fname))
		
		input_image = transform_image(image).unsqueeze(0).to('cuda')
		# Prediction
		with torch.no_grad():
			predictions = model(input_image)[-1].sigmoid().cpu()
			pred = predictions[0].squeeze()
			pred_pil = transforms.ToPILImage()(pred)
			mask = pred_pil.resize(image.size)
			mask.save(os.path.join(base_path, 'mask', f'mask_{fname}'))
