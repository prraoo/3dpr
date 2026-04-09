import os
import sys
import glob
from natsort import natsorted
import cv2
import numpy as np
import torch
import platform
import dlib
try:
	from evaluation.utils import resize_n_crop_numpy
except ModuleNotFoundError:
	from utils import resize_n_crop_numpy

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_path not in sys.path:
	sys.path.insert(0, base_path)
from magface.loss import MagFaceLoss, img2magfaceInput
magface = MagFaceLoss()
id_loss_fn = magface.build_model()

import lpips
lpips_loss_fn = lpips.LPIPS(net='alex').cuda()

from DISTS_pt import DISTS
dists_loss_fn = DISTS().cuda()

if os == 'windows':
	root = '/CT/HPS/'
else:
	root = '/HPS/'

if platform.system() == 'Windows':
	raise NotImplementedError
else:
	root = '/HPS/'
	predictor = dlib.shape_predictor('/home/prao/windows/nerf/preprocessing/shape_predictor_68_face_landmarks.dat')

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
			face_mask_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
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
			# IC-Light - MPI dataset
			base_dir          = '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test'
			landmarks_dir     = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			face_seg_mask_dir = '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/'
			model_dir         = '/CT/VORF_GAN4/nobackup/code/ic-light/'
		else:
			raise NotImplementedError

	# All input environment maps Loop:
	print(f"***********Evaluation of {expt_dir}******************")

	# Full Test Set
	CAM_LIST = ['Cam04', 'Cam06', 'Cam07', 'Cam23', 'Cam39']
	BASE_EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']
	EMAP_LIST = ['860', '861', '862', '863', '864', '865', '866', '867', '868', '869']

	lpips_scores = []
	rmse_scores = []
	dists_scores = []
	id_loss_scores = []
	landmarks_scores = []

	# For each input envmap, eval all cameras and all envmaps
	for BASE_EMAP in BASE_EMAP_LIST:
		pred_id_path = f'ID20{ID}_EMAP-{BASE_EMAP}'
		# print(pred_id_path)

		# For each input envmap, and every test envmap, eval all cameras
		for EMAP in EMAP_LIST:
			for CAM in CAM_LIST:

				if dir is not None:
					pred_path= dir
				else:
					pred_path = os.path.join(model_dir, pred_id_path, expt_dir)

					if baseline == 1:
						# 3DPR - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'evaluation-sigg25', pred_id_path)
						assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					elif baseline == 2:
						# Lite2Relight - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval-sigg25', pred_id_path)
						try:
							assert os.path.isdir(pred_path)
						except AssertionError as e:
							print(f'{e}, {pred_path} does not exist')
						# assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					elif baseline == 3:
						# NFL - MPI
						pred_path = os.path.join(model_dir, expt_dir, 'inversion-eval-sigg25', pred_id_path)
						assert os.path.isdir(pred_path), f'{pred_path} does not exist'
					elif baseline == 4:
						# IC-Light - MPI
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
					# IC-Light - MPI
					pred_images_SR = natsorted(glob.glob1(pred_path, f'{CAM}*{EMAP}.png'))
				else:
					raise NotImplementedError
				
				id_path = f'ID20{ID}'
				# Ground Truth
				gt_path = os.path.join(base_dir, id_path, 'images/')
				gt_images = natsorted(glob.glob1(gt_path, f'{CAM}*{EMAP}.png'))

				# Mask Path
				# mask_path = os.path.join(face_seg_mask_dir, id_path, 'mask_seg/')
				mask_path = os.path.join(face_seg_mask_dir, id_path, 'bgMatting/')
				assert os.path.isdir(mask_path), f'{mask_path} does not exist'
				mask_images = natsorted(glob.glob1(mask_path, f'{CAM}_{id_path}.png'))
				assert len(gt_images) == len(mask_images), f'Found {len(mask_images)} masks but only {len(gt_images)} images'

				# Landmarks
				landmarks_path = os.path.join(landmarks_dir, id_path, 'transform/')
				landmarks = natsorted(glob.glob1(landmarks_path, f'{CAM}_{id_path}.txt'))
				# print(landmarks)

				# Eye Mask - Needed for VoRF?
				eye_mask_path = os.path.join(landmarks_dir, id_path, 'mask/')
				eye_mask_images = natsorted(glob.glob1(eye_mask_path, f'{CAM}*.png'))

				## Evaluation Metrics

				for pred_img_SR_path, gt_img, mask, eye_mask, lm in zip(pred_images_SR, gt_images, mask_images, eye_mask_images, landmarks):

					# Landmarks based cropping
					gt_img_raw = cv2.imread(os.path.join(gt_path, gt_img))

					if gt_img_raw.shape[0] == 1030:
						gt_img_raw = cv2.rotate(gt_img_raw, cv2.ROTATE_90_CLOCKWISE)
					scale_crop_params = np.loadtxt(os.path.join(landmarks_path, lm))

					# mask = None
					mask = cv2.imread(os.path.join(mask_path, mask))/255

					gt_img = resize_n_crop_numpy(gt_img_raw, scale_crop_params)
					if mask.shape[0] == 4096:
						mask = resize_n_crop_numpy(mask, scale_crop_params)

					pred_img = cv2.imread(os.path.join(pred_path, pred_img_SR_path))
					if mask is not None:
						gt_img = gt_img*mask
						pred_img = pred_img * mask

					cv2.imwrite('debug.png', np.hstack([gt_img, pred_img]))
					
					# Compute Landmark Loss
					gray1 = cv2.resize((gt_img).astype(np.uint8), (128*2,128*2))
					gray2 = cv2.resize((pred_img).astype(np.uint8), (128*2, 128*2))
					
					# Detect facial landmarks in the images
					landmarks1 = predictor(gray1, dlib.rectangle(0, 0, gray1.shape[1], gray1.shape[0])).parts()
					landmarks2 = predictor(gray2, dlib.rectangle(0, 0, gray2.shape[1], gray2.shape[0])).parts()
					
					diff = 0
					for i, (point1, point2) in enumerate(zip(landmarks1, landmarks2)):
						diff += abs(point1.x - point2.x) + abs(point1.y - point2.y)
					landmarks_scores.append(diff)
				
					# Compute ID loss
					mf_pred_img = img2magfaceInput(pred_img)
					mf_gt_img = img2magfaceInput(gt_img)
					emb = id_loss_fn(torch.cat([mf_pred_img, mf_gt_img], dim=0))
					
					pred_emb = emb[0].data.cpu().numpy()
					pred_emb = pred_emb/ np.linalg.norm(pred_emb)

					gt_emb = emb[1].data.cpu().numpy()
					gt_emb = gt_emb/ np.linalg.norm(gt_emb)

					id_simialrity = np.dot(pred_emb, gt_emb.T)
					id_loss_scores.append(id_simialrity)

					
					gt_img = torch.Tensor((gt_img / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
					pred_img = torch.Tensor((pred_img / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()

					# Compute lpips distance
					dist_lpips = lpips_loss_fn.forward(gt_img, pred_img)
					lpips_scores.append(dist_lpips.data.cpu().numpy())

					# Compute RMSE distance
					dist_rmse = torch.sqrt(torch.mean((gt_img - pred_img) ** 2))
					rmse_scores.append(dist_rmse.data.cpu().numpy())

					# Compute DISTS distance
					error_dists = dists_loss_fn.forward(gt_img, pred_img)
					dists_scores.append(error_dists.data.cpu().numpy())


	print(f'Evaluated a total of {len(dists_scores)}')
	print(f'Average LPIPS   = {np.mean(lpips_scores)}')
	print(f'Average RMSE    = {np.mean(rmse_scores)}')
	print(f'Average DISTS   = {np.mean(dists_scores)}')
	print(f'Average ID SIM  = {np.mean(id_loss_scores)}')
	print(f'Average LD  = {np.mean(landmarks_scores)/68.}')

	return np.mean(lpips_scores), np.mean(rmse_scores), np.mean(dists_scores), np.mean(id_loss_scores), np.mean(landmarks_scores)


if __name__ == '__main__':
	scores_lpips_list = []
	scores_rmse_list = []
	scores_dists_list = []
	scores_id_loss_list = []
	scores_landmarks_list = []
	
	identity_list = ['ID20029', 'ID20031', 'ID20035', 'ID20037', 'ID20045', 'ID20047', 'ID20059', 'ID20090', 'ID20104', 'ID20113']
	baseline_method = int(sys.argv[1])

	for identity in identity_list:
		print(f"{identity}")
		l, r, d, id, ld = run_eval(identity=identity, baseline=baseline_method)
		scores_lpips_list.append(l)
		scores_rmse_list.append(r)
		scores_dists_list.append(d)
		scores_id_loss_list.append(id)
		scores_landmarks_list.append(ld)
		
	print(f'\n Final Averages \n')
	print(f'Average LPIPS = {np.mean(scores_lpips_list)}')
	print(f'Average RMSE =  {np.mean(scores_rmse_list)}')
	print(f'Average DISTS = {np.mean(scores_dists_list)}')
	print(f'Average ID SIM = {np.mean(scores_id_loss_list)}')
	print(f'Average LD = {np.mean(scores_landmarks_list)/68.}')

