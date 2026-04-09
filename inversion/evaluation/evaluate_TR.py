import os
import glob
from natsort import natsorted
import cv2
import numpy as np
import torch
import lpips
from DISTS_pt import DISTS
from image_metric import PSNR, SSIM

lpips_loss_fn = lpips.LPIPS(net='alex').cuda()
dists_loss_fn = DISTS().cuda()
psnr_score_fn = PSNR()
ssim_score_fn = SSIM()


def crop_resize_image(image):
	assert image.shape[1] == 768 and image.shape[0] == 1024

	# Calculate top and bottom cropping to make the image square 768x768
	top_crop = (1024 - 768) // 2
	bottom_crop = 1024 - top_crop

	# Crop the image to make it square
	cropped_image = image[top_crop:bottom_crop, 0:768]

	# Resize the square image to 512x512
	final_image = cv2.resize(cropped_image, (512, 512))

	return final_image

if __name__ == '__main__':
	# Define all the paths

	face_seg_mask_dir = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/'
	OURS_BASE_PATH = '/CT/VORF_GAN3/nobackup/code/goae-inversion-olat/training-debug/'

	# EXP_DIR        = '03_mrf_loss-abln/00010-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0/'
	EXP_DIR        = '03_mrf_loss-abln/00035-gpus4-batch8-250ids-32o_dim--1_lpips0.05-lr0.0001-ldr1-L1-disc0-g_feat0/'


	TR_BASE_PATH   = '/CT/VORF_GAN3/work/code/TotalRelighting/'
	EMAP_PATH      =  os.path.join(TR_BASE_PATH, 'envs')

	MODE = 0  # 0: lightstage or 1: in-the-wild
	if MODE == 0:
		OURS_PRED_PATH = os.path.join(OURS_BASE_PATH, EXP_DIR, 'evaluation-TotalRelighting')
		# TR_PRED_PATH = os.path.join(TR_BASE_PATH, '00_output')
		TR_PRED_PATH = os.path.join(TR_BASE_PATH, '00_output_180')
		TR_PRED_PATH = os.path.join(TR_BASE_PATH, '00_output_mirrored')
		GT_PATH = os.path.join(TR_BASE_PATH, '00_gt')
		get_scores = True
	else:
		OURS_PRED_PATH = os.path.join(OURS_BASE_PATH, EXP_DIR, 'evaluation-TotalRelighting-wild')
		# TR_PRED_PATH = os.path.join(TR_BASE_PATH, '01_output')
		# TR_PRED_PATH = os.path.join(TR_BASE_PATH, '01_output_180')
		TR_PRED_PATH = os.path.join(TR_BASE_PATH, '01_output_mirrored')
		GT_PATH = os.path.join(TR_BASE_PATH, '00_gt')
		get_scores = False
		ref_scan_name = 'ID00119'


	# save_dir = os.path.join(TR_BASE_PATH, f'{MODE:02d}_combined')
	# save_dir = os.path.join(TR_BASE_PATH, f'{MODE:02d}_combined_180')
	save_dir = os.path.join(TR_BASE_PATH, f'{MODE:02d}_combined_mirrored')
	os.makedirs(save_dir, exist_ok=True)
	scan_names_list = [dir for dir in natsorted(glob.glob1(OURS_PRED_PATH, '*'))]
	emap_list = [emap for emap in natsorted(glob.glob1(EMAP_PATH, '*.exr'))]

	ssim_scores  = []
	psnr_scores  = []
	lpips_scores = []
	rmse_scores  = []
	dists_scores = []
	tr_ssim_scores  = []
	tr_psnr_scores  = []
	tr_lpips_scores = []
	tr_rmse_scores  = []
	tr_dists_scores = []
	if MODE == 1:
		assert get_scores is False

	save_fig = True

	for scan_name_dir in scan_names_list[0:]:
		scan_name = scan_name_dir.split('_')[0]
		print(f"Evaluating Idenitity {scan_name} ... ")

		#TODO: loop over all the emaps
		for emap_name in emap_list[:]:
			emap_name = emap_name[:-4]

			# TODO: load TR images
			tr_image_path = os.path.join(TR_PRED_PATH, f'{scan_name}_{emap_name}.png')
			assert os.path.isfile(tr_image_path), f'{tr_image_path} not found!'
			tr_image_raw = cv2.imread(tr_image_path)
			# TODO: resize TR images to 512 x 512
			tr_image = crop_resize_image(tr_image_raw)

			# TODO: load gt
			if MODE == 0:
				gt_image_path = os.path.join(GT_PATH, f'{scan_name}_{emap_name}.png')
				assert os.path.isfile(gt_image_path), f'{gt_image_path} not found!'
				gt_image = cv2.imread(gt_image_path)
			else:
				gt_image_path = os.path.join(GT_PATH, f'{ref_scan_name}_{emap_name}.png')
				assert os.path.isfile(gt_image_path), f'{gt_image_path} not found!'
				gt_image = cv2.imread(gt_image_path)


			# TODO: load ours
			try:
				ours_image_path = os.path.join(OURS_PRED_PATH, scan_name_dir, f'Cam07_{scan_name}_{emap_name}.png')
				assert os.path.isfile(ours_image_path)
			except AssertionError:
				try:
					ours_image_path = os.path.join(OURS_PRED_PATH, scan_name_dir, f'Cam07_{scan_name}_{emap_name.replace("EMAP-", "EMAP_")}.png')
					assert os.path.isfile(ours_image_path), f'{ours_image_path} not found!'
				except:
					ours_image_path = os.path.join(OURS_PRED_PATH, scan_name_dir, f'{scan_name}_{emap_name.replace("EMAP-", "EMAP_")}.png')
					assert os.path.isfile(ours_image_path), f'{ours_image_path} not found!'

			ours_image = cv2.imread(ours_image_path)

			# save concat results
			if save_fig is True:
				#TODO: add downsampled env map below
				gt_image_save = gt_image.copy()
				emap_png = cv2.imread(os.path.join(EMAP_PATH, f'{emap_name}.png'))
				scale = 8
				image_20x40 = cv2.resize(emap_png, (20*scale, 10*scale), interpolation=cv2.INTER_AREA)
				# Determine the position to place the 20x40 image in the lower right corner
				x_offset = 512 - 20*scale  # Calculate the x-offset for the lower right corner
				y_offset = 512 - 10*scale  # Calculate the y-offset for the lower right corner

				# Copy the 20x40 image into the lower right corner of the 512x512 image
				gt_image_save[y_offset:y_offset + 10*scale, x_offset:x_offset + 20*scale] = image_20x40
				cv2.imwrite(os.path.join(save_dir, f'{scan_name}_{emap_name}.png'), cv2.hconcat([gt_image_save, ours_image, tr_image]))




			if get_scores is True:
				# TODO: load masks
				mask_path = os.path.join(face_seg_mask_dir, scan_name, 'mask_seg/')
				mask_images = natsorted(glob.glob1(mask_path, f'Cam07_{scan_name}_*.png'))
				assert len(mask_images) == 1
				mask_image = cv2.imread(os.path.join(mask_path, mask_images[0])) / 255

				# loss metrics
				gt_img = torch.Tensor((gt_image / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
				pred_img = torch.Tensor((ours_image / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
				tr_pred_img = torch.Tensor((tr_image / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()

				# Compute lpips distance
				dist_lpips = lpips_loss_fn.forward(gt_img, pred_img)
				lpips_scores.append(dist_lpips.data.cpu().numpy())

				tr_dist_lpips = lpips_loss_fn.forward(gt_img, tr_pred_img)
				tr_lpips_scores.append(tr_dist_lpips.data.cpu().numpy())

				# Compute lpips distance
				dist_rmse = torch.sqrt(torch.mean((gt_img - pred_img) ** 2))
				rmse_scores.append(dist_rmse.data.cpu().numpy())

				tr_dist_rmse = torch.sqrt(torch.mean((gt_img - tr_pred_img) ** 2))
				tr_rmse_scores.append(tr_dist_rmse.data.cpu().numpy())

				# Compute DISTS distance
				error_dists = dists_loss_fn.forward(gt_img, pred_img)
				dists_scores.append(error_dists.data.cpu().numpy())

				tr_error_dists = dists_loss_fn.forward(gt_img, tr_pred_img)
				tr_dists_scores.append(tr_error_dists.data.cpu().numpy())

				# Compute PSNR distance
				score_psnr = psnr_score_fn.compute(gt_image, ours_image)
				psnr_scores.append(score_psnr)

				tr_score_psnr = psnr_score_fn.compute(gt_image, tr_image)
				tr_psnr_scores.append(tr_score_psnr)

				# Compute SSIM distance
				score_ssim = ssim_score_fn.compute(gt_image*mask_image, ours_image*mask_image)
				ssim_scores.append(score_ssim)

				tr_score_ssim = ssim_score_fn.compute(gt_image*mask_image, tr_image*mask_image)
				tr_ssim_scores.append(tr_score_ssim)

	if get_scores:
		print(f'Evaluated a total of {len(dists_scores)}')
		print(f'Average LPIPS = Ours: {np.mean(lpips_scores):0.4f} | TR: {np.mean(tr_lpips_scores):0.4f} | TR: {np.mean(sorted(tr_lpips_scores, reverse=True)[:150]):0.4f} ')
		print(f'Average RMSE =  Ours: {np.mean(rmse_scores):0.4f}  | TR: {np.mean(tr_rmse_scores):0.4f}  | TR: {np.mean(sorted(tr_rmse_scores, reverse=True)[:150]) :0.4f} ')
		print(f'Average DISTS = Ours: {np.mean(dists_scores):0.4f} | TR: {np.mean(tr_dists_scores):0.4f} | TR: {np.mean(sorted(tr_dists_scores, reverse=True)[:150]):0.4f} ')
		print(f'Average PSNR =  Ours:  {np.mean(psnr_scores):0.4f} | TR: {np.mean(tr_psnr_scores):0.4f}  | TR: {np.mean(sorted(tr_psnr_scores, reverse=True)[:150]) :0.4f} ')
		print(f'Average SSIM =  Ours:  {np.mean(ssim_scores):0.4f} | TR: {np.mean(tr_ssim_scores):0.4f}  | TR: {np.mean(sorted(tr_ssim_scores, reverse=False)[:150]) :0.4f}')








