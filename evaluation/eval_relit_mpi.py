import os
import glob
import sys

from natsort import natsorted
import cv2
import numpy as np
import platform
try:
	from evaluation.image_metric import PSNR, SSIM
	from evaluation.utils import resize_n_crop_numpy
except ModuleNotFoundError:
	from image_metric import PSNR, SSIM
	from utils import resize_n_crop_numpy
if os == 'windows':
	root = '/CT/HPS/'
else:
	root = '/HPS/'


def run_eval(dir=None, identity='', baseline=0):
	if len(identity) == 7:
		ID = identity[-3:]
	else:
		ID = 307

	if baseline == 1:
	# 3DPR - MPII dataset
		expt_dir = '00049-gpus3-batch9-450ids-1views-resnet-half_res--1_lpips0.3-L1_loss-lcol1.0-lr0.00015'
	elif baseline == 2:
		# Lite2Relight - MPII dataset
		expt_dir = '00002-gpus2-batch8-real-resnet-lpips1.0-lr0.0003-ldr1-disc0'
	elif baseline == 3:
		# # NFL - MPII dataset
		expt_dir = ''
	elif baseline == 4:
		# IC-Light
		expt_dir = '00049-gpus3-batch9-450ids-1views-resnet-half_res--1_lpips0.3-L1_loss-lcol1.0-lr0.00015'
	else:
		raise NotImplementedError

	print("Evaluating: ", expt_dir)

	## Ground Truth
	if platform.system() == 'Windows':
		raise NotImplementedError
	else:
		if baseline == 1:
			# 3DPR - MPI dataset
			base_dir          = '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test'
			landmarks_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			face_seg_mask_dir = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			model_dir = '/CT/VORF_GAN4/nobackup/code/goae-inversion-olat/training-debug/04_runs/'
		elif baseline == 2:
			# Lite2Relight - MPI dataset
			base_dir          = '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test'
			landmarks_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			face_seg_mask_dir = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			model_dir         = '/CT/VORF_GAN3/nobackup/code/goae-inversion-photoapp/training-runs/'
		elif baseline == 3:
			# NFL - MPI dataset
			base_dir          = '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test'
			landmarks_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			face_seg_mask_dir = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			model_dir         = '/CT/VORF_GAN3/nobackup/code/NerfFaceLighting/'
		elif baseline == 4:
			# IC-Light
			base_dir          = '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test'
			landmarks_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			face_seg_mask_dir = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			model_dir         = '/CT/VORF_GAN4/nobackup/code/ic-light/'
		else:
			raise NotImplementedError


	# All input environment maps Loop:

	# Full Test Set
	CAM_LIST = ['Cam04', 'Cam06', 'Cam07', 'Cam23', 'Cam39']
	BASE_EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']
	EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']

	all_input_psnrs_all_cams_all_env_m = []
	all_input_ssims_all_cams_all_env_m = []

	# For each input envmap, eval all cameras and all envmaps
	base_env_count = 0
	for BASE_EMAP in BASE_EMAP_LIST:
		base_env_count += 1
		psnrs_all_cams_all_env = 0
		ssims_all_cams_all_env = 0

		psnrs_all_cams_all_env_m = []
		ssims_all_cams_all_env_m = []

		pred_id_path = f'ID20{ID}_EMAP-{BASE_EMAP}'
		count_emap = 0
		# For each input envmap, and every test envmap, eval all cameras
		for EMAP in EMAP_LIST:
			psnrs_all_cams = 0
			ssims_all_cams = 0
			count_emap += 1
			
			for CAM in CAM_LIST:
				

				if dir is not None:
					pred_path= dir
				else:
					if baseline == 1:
						# 3DPR - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'evaluation-sigg25', pred_id_path)
						# pred_path = os.path.join(model_dir, expt_dir, 'evaluation-siga25-rotated', pred_id_path)
						assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					elif baseline == 2:
						# Lite2Relight - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval-sigg25', pred_id_path)
						try:
							assert os.path.isdir(pred_path)
						except AssertionError as e:
							print(f'{e}, {pred_path} does not exist')

					elif baseline == 3:
						# NFL - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval-sigg25', pred_id_path)
						assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					elif baseline == 4:
						# IC-Light
						pred_path = os.path.join(model_dir, expt_dir, 'evaluation-quantitative', pred_id_path)
						assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					else:
						raise NotImplementedError

				if baseline == 1:
					# 3DPR - MPI
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))
				
				elif baseline == 2:
					# Lite2Relight - MPI
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}_SR.png'))
				
				elif baseline == 3:
					# NeRFFaceLighting
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))
				elif baseline == 4:
					# IC-Light
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))
				else:
					raise NotImplementedError
				
				id_path = f'ID20{ID}'
				# Ground Truth
				gt_path = os.path.join(base_dir, id_path, 'images/')
				assert os.path.isdir(gt_path), f'{gt_path} does not exist'
				gt_images = natsorted(glob.glob1(gt_path, f'{CAM}*{EMAP}.png'))
				

				# Mask Path
				mask_path = os.path.join(face_seg_mask_dir, id_path, 'bgMatting/')
				assert os.path.isdir(mask_path), f'{mask_path} does not exist'
				mask_images = natsorted(glob.glob1(mask_path, f'{CAM}_{id_path}.png'))
				
				# Landmarks
				landmarks_path = os.path.join(landmarks_dir, id_path, 'transform/')
				landmarks = natsorted(glob.glob1(landmarks_path, f'{CAM}_{id_path}.txt'))

				# Eye Mask # Needed for VoRF?
				eye_mask_path = os.path.join(landmarks_dir, id_path, 'mask/')
				eye_mask_images = natsorted(glob.glob1(eye_mask_path, f'{CAM}*.png'))

				## Evaluation Metrics
				psnr_calc, ssim_calc = PSNR(), SSIM()
				psnrs = 0
				ssims = 0
				count = 0
				for pred_img_SR_path, gt_img, mask, eye_mask, lm in zip(pred_images_SR, gt_images, mask_images, eye_mask_images, landmarks):
					count += 1
					# Landmarks based cropping
					gt_img_raw = cv2.imread(os.path.join(gt_path, gt_img))
					if gt_img_raw.shape[0] == 1030:
						gt_img_raw = cv2.rotate(gt_img_raw, cv2.ROTATE_90_CLOCKWISE)

					# gt_img_raw = cv2.rotate(cv2.imread(os.path.join(gt_path, gt_img)), cv2.ROTATE_90_CLOCKWISE)
					# mask_raw = cv2.rotate(cv2.imread(os.path.join(mask_path, mask)), cv2.ROTATE_90_CLOCKWISE)/255
					mask = cv2.imread(os.path.join(mask_path, mask))/255
					scale_crop_params = np.loadtxt(os.path.join(landmarks_path, lm))

					# TODO: based on landmarks, crop eyes for VoRF
					# eye_mask = cv2.imread(os.path.join(eye_mask_path,eye_mask))

					gt_img = resize_n_crop_numpy(gt_img_raw, scale_crop_params)
					if mask.shape[0] == 4096:
						mask = resize_n_crop_numpy(mask, scale_crop_params)
					pred_img_SR = cv2.imread(os.path.join(pred_path, pred_img_SR_path))

					# 3DPR-New
					_ssims = ssim_calc.compute(gt_img * mask, pred_img_SR * mask)
					_psnrs = psnr_calc.compute(gt_img*mask, pred_img_SR*mask)

					# Per Camera
					ssims += _ssims
					psnrs += _psnrs
					# All Cameras
					ssims_all_cams += _ssims
					psnrs_all_cams += _psnrs
					# All Cameras All Env Maps
					ssims_all_cams_all_env += _ssims
					psnrs_all_cams_all_env += _psnrs
					##
					ssims_all_cams_all_env_m.append(_ssims)
					psnrs_all_cams_all_env_m.append(_psnrs)

					cv2.imwrite('test_pred.png', pred_img_SR*mask)

		all_input_ssims_all_cams_all_env_m.extend(ssims_all_cams_all_env_m)
		all_input_psnrs_all_cams_all_env_m.extend(psnrs_all_cams_all_env_m)
	print(f'---------------------{ID}-----------------------')
	print(f'{ID}-Avg: SSIM, PSNR = {np.median(all_input_ssims_all_cams_all_env_m):0.4f}',
	      f'\t{np.median(all_input_psnrs_all_cams_all_env_m):0.4f}',
	      f'\t{np.mean(all_input_ssims_all_cams_all_env_m):0.4f}'
	      f'\t{np.mean(all_input_psnrs_all_cams_all_env_m):0.4f}'

	      )
	print(f'--------------------------------------------')
	
	return np.median(all_input_ssims_all_cams_all_env_m), np.median(all_input_psnrs_all_cams_all_env_m)



if __name__ == '__main__':
	identity_list = ['ID20029', 'ID20031', 'ID20035', 'ID20037', 'ID20045', 'ID20047', 'ID20059', 'ID20090', 'ID20104', 'ID20113']
	baseline_method = int(sys.argv[1])
	scores_psnr_list = []
	scores_ssim_list = []
	for identity in identity_list:
		score_ssim, score_psnr = run_eval(identity=identity, baseline=baseline_method)
		scores_ssim_list.append(score_ssim)
		scores_psnr_list.append(score_psnr)
		
	print(f'\n Final Averages \n')
	print(f'Average SSIM = {np.mean(scores_ssim_list)}')
	print(f'Average PSNR = {np.mean(scores_psnr_list)}')
