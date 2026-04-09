import os
import glob
from natsort import natsorted
import cv2
import numpy as np
import dlib

# import torch
# from torch.nn import functional as F
try:
	from evaluation.image_metric import PSNR, SSIM
	from evaluation.utils import resize_n_crop_numpy
except ModuleNotFoundError:
	from image_metric import PSNR, SSIM
	from utils import resize_n_crop_numpy


import platform
if platform.system() == 'Windows':
	root = '/CT/HPS/'
	# Load the pre-trained facial landmark detector from dlib
	predictor = dlib.shape_predictor("W:\\nerf\\preprocessing\\shape_predictor_68_face_landmarks.dat")
else:
	root = '/HPS/'
	predictor = dlib.shape_predictor('/home/prao/windows/nerf/preprocessing/shape_predictor_68_face_landmarks.dat')



def run_eval(dir=None, identity=''):
	if len(identity) == 7:
		ID = identity[-3:]
	else:
		ID = 307

	## Predictions
	# expt_dir = '312IDs_CNN_stage2_CNN_1resblk_3MLP_no_disc_multiview_150-OLAT_dirs-25_multi_envmap_HDR_LPIPS_norm-adv-021'

	# RelightingInTheWild
	expt_dir = '00002-gpus2-batch8-real-resnet-lpips1.0-lr0.0003-ldr1-disc0'

	# Main Model - 1
	# expt_dir = '00032-gpus2-batch08-id100-mse0.0-lpips1.0-id0.0-latent10.0-lr0.0001-norm0-tmap0-emaps_ds'
	# expt_dir = '00044-gpus2-batch08-id250-mse0.1-lpips1.0-id0.0-latent10.0-lr0.0003-emaps_ds-Cam2'
	# expt_dir = '00030-gpus2-batch08-id250-mse0.0-lpips1.0-id0.0-latent10.0-lr0.0003-norm0-tmap0-emaps_ds'
	# Ablation - LPIPS
	# expt_dir = '00045-gpus2-batch08-id250-mse0.0-lpips0.0-id0.0-latent10.0-lr0.0003-emaps_ds-Cam2'

	# Ablation - Num Subjects
	# expt_dir = '00032-gpus2-batch08-id10-mse0.0-lpips1.0-id0.0-latent10.0-lr0.0001-norm0-tmap0-emaps_ds'
	# expt_dir = '00032-gpus2-batch08-id50-mse0.0-lpips1.0-id0.0-latent10.0-lr0.0001-norm0-tmap0-emaps_ds'

	# Ablations: Num cams
	# expt_dir = '00037-gpus2-batch08-id250-mse0.0-lpips0.0-id0.0-latent10.0-lr0.0001-emaps_ds-Cam8'
	# expt_dir = '00038-gpus2-batch08-id250-mse0.0-lpips0.0-id0.0-latent10.0-lr0.0001-emaps_ds-Cam4'
	# expt_dir = '00036-gpus2-batch08-id250-mse0.0-lpips0.0-id0.0-latent10.0-lr0.0001-emaps_ds-Cam1'


	# Ablation - F Space
	# expt_dir = '00032-gpus2-batch08-id100-mse0.0-lpips1.0-id0.0-latent10.0-lr0.0001-norm0-tmap0-emaps_ds-no_feat'

	# ----------------------------------------------------------------------
	# 3DPR - New
	# Main Model
	expt_dir = '00010-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00021-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00035-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'

	# Abln - Losses
	# expt_dir = '00019-gpus4-batch8-250ids-32o_dim-0_lpips0.0-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00020-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00008-gpus4-batch16-250ids-12o_dim-2_lpips10.0-lr0.0001-ldr1-L1-disc0-g_feat0'

	# Abln - Feature Encoder
	# expt_dir = '00004-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'

	# Abln - N Subjects
	# expt_dir = '00015-gpus4-batch8-100ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00016-gpus4-batch8-50ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00034-gpus4-batch8-5ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'

	## Abln - N Dim
	# expt_dir = '00011-gpus4-batch8-250ids-16o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00010-gpus4-batch12-250ids-12o_dim--1_lpips0.025-lr0.0001-ldr1-L1-disc0-g_feat0'
	# expt_dir = '00010-gpus4-batch16-250ids-3o_dim-2_lpips10.0-lr0.0001-ldr1-L1-disc0-g_feat0'

	# NeRFFaceLighting
	# expt_dir = ''

	print("Evaluating: ", expt_dir)

	## Ground Truth
	if platform.system() == 'Windows':
		base_dir = '//winfs-inf/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-test/'
		landmarks_dir = '//winfs-inf/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		face_seg_mask_dir = '//winfs-inf/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		face_mask_dir = '//winfs-inf/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/metashape-300IDs/10001/'
		model_dir = '//winfs-inf/CT/VORF_GAN2/work/EG3D/stage2_no_PTI_SR/inversion/'
	else:
		base_dir = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-test/'
		# base_dir = '/CT/LS_FRM02/static00/datasets/OLAT_c2-Multiple-IDs-eval/eval/'
		landmarks_dir = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		face_seg_mask_dir = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
		face_mask_dir = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/metashape-300IDs/10001/'
		model_dir = '/CT/VORF_GAN2/work/EG3D/stage2_no_PTI_SR/inversion/'

		# Lite2Relight
		model_dir = '/CT/VORF_GAN3/nobackup/code/goae-inversion-photoapp/training-runs/'

		# 3DPR - New
		model_dir = '/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/training-debug/03_mrf_loss-abln/'

		# #NerFaceLighting
		# model_dir = '/CT/VORF_GAN3/nobackup/code/NerfFaceLighting/'
	# All input environment maps Loop:

	# Full Test Set
	# CAM_LIST = ['Cam07']
	CAM_LIST = ['Cam05', 'Cam06', 'Cam07', 'Cam08', 'Cam10', 'Cam15']
	BASE_EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']
	# BASE_EMAP_LIST = ['860']
	EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']
	# EMAP_LIST = ['860']

	BASE_EMAP_LIST = ['860']
	# CAM_LIST = ['Cam06', 'Cam07', 'Cam15']
	CAM_LIST = ['Cam05', 'Cam06', 'Cam07', 'Cam08', 'Cam10', 'Cam15']
	# EMAP_LIST = [f'{860+x}' for x in range(20)]
	EMAP_LIST = [f'{860+x}' for x in range(10)]

	all_input_lm_loss_all_cams_all_env = 0
	all_input_psnrs_all_cams_all_env = 0
	all_input_ssims_all_cams_all_env = 0

	all_input_lm_loss_all_cams_all_env_m = []
	all_input_psnrs_all_cams_all_env_m = []
	all_input_ssims_all_cams_all_env_m = []

	# For each input envmap, eval all cameras and all envmaps
	for BASE_EMAP in BASE_EMAP_LIST:
		lm_loss_all_cams_all_env = 0
		psnrs_all_cams_all_env = 0
		ssims_all_cams_all_env = 0

		lm_loss_all_cams_all_env_m = []
		psnrs_all_cams_all_env_m = []
		ssims_all_cams_all_env_m = []

		pred_id_path = f'ID00{ID}_EMAP-{BASE_EMAP}'
		print(pred_id_path)

		# For each input envmap, and every test envmap, eval all cameras
		for EMAP in EMAP_LIST:
			lm_loss_all_cams = 0
			psnrs_all_cams = 0
			ssims_all_cams = 0

			for CAM in CAM_LIST:

				if dir is not None:
					pred_path= dir
				else:
					pred_path = os.path.join(model_dir, pred_id_path, expt_dir)

					# RelightingInTheWild
					# pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval-main', pred_id_path)
					# pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval', pred_id_path)
					# pred_path = os.path.join(model_dir, expt_dir, 'evaluation-siga24-diversity', pred_id_path)

					# pred_images = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))
					
					# pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}_SR.png'))
					# print(pred_id_path, pred_images_SR)
					
					# 3DPR - New
					pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval', pred_id_path)
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))


				id_path = f'ID00{ID}'
				# Ground Truth
				gt_path = os.path.join(base_dir, id_path, 'images/')
				gt_images = natsorted(glob.glob1(gt_path, f'{CAM}*{EMAP}.png'))
				# print(gt_images)

				# Mask Path
				# mask_path = os.path.join(face_mask_dir, id_path, 'mask_seg/')
				mask_path = os.path.join(face_seg_mask_dir, id_path, 'mask_seg/')
				mask_images = natsorted(glob.glob1(mask_path, f'{CAM}_{id_path}_*.png'))
				# print(mask_images)

				# Landmarks
				landmarks_path = os.path.join(landmarks_dir, id_path, 'transform/')
				landmarks = natsorted(glob.glob1(landmarks_path, f'{CAM}_{id_path}.txt'))
				# print(landmarks)

				# Eye Mask
				eye_mask_path = os.path.join(landmarks_dir, id_path, 'mask/')
				eye_mask_images = natsorted(glob.glob1(eye_mask_path, f'{CAM}*.png'))

				## Evaluation Metrics
				# psnr_calc, ssim_calc = PSNR(), SSIM()
				# psnrs = 0
				# ssims = 0
				lm_loss = 0

				# for pred_img, pred_img_SR, gt_img, mask, eye_mask, lm in zip(pred_images, pred_images_SR, gt_images, mask_images, eye_mask_images, landmarks):
				for pred_img_SR_path, gt_img, mask, eye_mask, lm in zip(pred_images_SR, gt_images, mask_images, eye_mask_images, landmarks):

					# Landmarks based cropping
					gt_img_raw = cv2.imread(os.path.join(gt_path, gt_img))
					if gt_img_raw.shape[0] == 1030:
						gt_img_raw = cv2.rotate(gt_img_raw, cv2.ROTATE_90_CLOCKWISE)
					# gt_img_raw = cv2.rotate(cv2.imread(os.path.join(gt_path, gt_img)), cv2.ROTATE_90_CLOCKWISE)
					scale_crop_params = np.loadtxt(os.path.join(landmarks_path, lm))
					mask = cv2.imread(os.path.join(mask_path, mask))/255

					gt_img = resize_n_crop_numpy(gt_img_raw, scale_crop_params)
					pred_img_SR = cv2.imread(os.path.join(pred_path, pred_img_SR_path))


					# No Masking
					# gray1 = cv2.resize(gt_img, (128,128))
					# gray2 = cv2.resize(pred_img_SR, (128, 128))

					# with mask
					gray1 = cv2.resize((gt_img*mask).astype(np.uint8), (128*2,128*2))
					gray2 = cv2.resize((pred_img_SR*mask).astype(np.uint8), (128*2, 128*2))

					# Detect facial landmarks in the images
					landmarks1 = predictor(gray1, dlib.rectangle(0, 0, gray1.shape[1], gray1.shape[0])).parts()
					landmarks2 = predictor(gray2, dlib.rectangle(0, 0, gray2.shape[1], gray2.shape[0])).parts()

					diff = 0
					for i, (point1, point2) in enumerate(zip(landmarks1, landmarks2)):
						diff += abs(point1.x-point2.x) + abs(point1.y-point2.y)

					# print(f'{CAM} - {diff}')

					# With Background Mask
					# if 'Cam05' not in pred_img_SR_path:
					# 	_ssims = ssim_calc.compute(gt_img * mask, pred_img_SR*mask)
					# 	_psnrs = psnr_calc.compute(gt_img * mask, pred_img_SR*mask)
					# else:
					# 	# Without Background Mask
					# 	_ssims = ssim_calc.compute(gt_img, pred_img_SR)
					# 	_psnrs = psnr_calc.compute(gt_img, pred_img_SR)

					# Per Camera
					# ssims += _ssims
					# psnrs += _psnrs
					lm_loss += diff
					# All Cameras
					# ssims_all_cams += _ssims
					# psnrs_all_cams += _psnrs
					lm_loss_all_cams += diff
					# All Cameras All Env Maps
					lm_loss_all_cams_all_env += diff
					# ssims_all_cams_all_env += _ssims
					# psnrs_all_cams_all_env += _psnrs
					##
					lm_loss_all_cams_all_env_m.append(diff)
					# ssims_all_cams_all_env_m.append(_ssims)
					# psnrs_all_cams_all_env_m.append(_psnrs)
					# print(len(ssims_all_cams_all_env_m), len(psnrs_all_cams_all_env_m)) # sanity check for number of images

					# cv2.imwrite('test_gt.png', gt_img*mask)
					# cv2.imwrite('test_pred.png', pred_img_SR*mask)

					# Low Resolution
						# gt_img = cv2.resize(gt_img, (128, 128))
						# ssims += ssim_calc.compute(gt_img, pred_img)
						# psnrs += psnr_calc.compute(gt_img, pred_img)
						#
						# ssims += ssim_calc.compute(gt_img * mask, pred_img * mask)
						# psnrs += psnr_calc.compute(gt_img * mask, pred_img * mask)


				# print(f'{CAM} SSIM = {ssims/len(gt_images):0.4f}')
				# print(f'{CAM} PSNR = {psnrs/len(gt_images):0.4f}')
		#
		# 	print(f'{EMAP} Average: SSIM, PSNR = {ssims_all_cams/(len(gt_images)*len(CAM_LIST)):0.4f}',
		# 		  f'{psnrs_all_cams/(len(gt_images)*len(CAM_LIST)):0.4f}')
		#
		# print(f'\nAverage: SSIM, PSNR = {ssims_all_cams_all_env/(len(gt_images)*len(CAM_LIST)*len(EMAP_LIST)):0.4f}'
		# 	  , f'\t{psnrs_all_cams_all_env/(len(gt_images)*len(CAM_LIST)*len(EMAP_LIST)):0.4f}')
		#
		# print(f'{ID}-Avg-{BASE_EMAP}: SSIM, PSNR = {np.median(ssims_all_cams_all_env_m):0.4f}', f'\t{np.median(psnrs_all_cams_all_env_m):0.4f}')
		# print(f'{ID}-Avg-{BASE_EMAP}: LM Loss = {np.median(lm_loss_all_cams_all_env_m):0.4f}')
		all_input_lm_loss_all_cams_all_env_m.extend(lm_loss_all_cams_all_env_m)
		# all_input_ssims_all_cams_all_env_m.extend(ssims_all_cams_all_env_m)
		# all_input_psnrs_all_cams_all_env_m.extend(psnrs_all_cams_all_env_m)

	print(f'---------------------{ID}-----------------------')
	# print(f'{ID}-Avg: SSIM, PSNR = {np.median(all_input_ssims_all_cams_all_env_m):0.4f}', f'\t{np.median(all_input_psnrs_all_cams_all_env_m):0.4f}')
	print(f'{ID}-Avg: LM Loss = {np.median(all_input_lm_loss_all_cams_all_env_m)/68.:0.4f}')
	print(f'--------------------------------------------')
	return np.median(all_input_lm_loss_all_cams_all_env_m)



if __name__ == '__main__':
	# identity_list = ['ID00307']
	identity_list = ['ID00307', 'ID00317','ID00333', 'ID00334','ID00337', 'ID00347','ID00348', 'ID00350', 'ID00352']
	# identity_list = [ 'ID00259', 'ID00300','ID00302', 'ID00333'] # Diversity Eval
	lm_scores = []
	for identity in identity_list:
		lm_score = run_eval(identity=identity)
		lm_scores.append(lm_score)

	print(lm_scores)

	print("Final Score: ", np.mean(lm_scores)/68)