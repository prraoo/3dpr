import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import dlib
import cv2
from PIL import Image

from segment_anything.sam2.build_sam import build_sam2
from segment_anything.sam2.sam2_image_predictor import SAM2ImagePredictor

np.random.seed(3)
# select the device for computation
if torch.cuda.is_available():
	device = torch.device("cuda")
elif torch.backends.mps.is_available():
	device = torch.device("mps")
else:
	device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
	# use bfloat16 for the entire notebook
	torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
	# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
	if torch.cuda.get_device_properties(0).major >= 8:
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
else:
	RuntimeError(f"Unsupported device: {device}")


def show_mask(mask, ax, obj_id=None, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		cmap = plt.get_cmap("tab10")
		cmap_idx = 0 if obj_id is None else obj_id
		color = np.array([*cmap(cmap_idx)[:3], 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
	pos_points = coords[labels == 1]
	neg_points = coords[labels == 0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# # TODO: Load all the images
# video_dir = './eval/data_video'
# # scan all the JPEG frame names in this directory
# frame_names = [
# 	p for p in os.listdir(video_dir)
# 	if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
# ]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
# print(frame_names)
#
# TODO: Build the model
from segment_anything.sam2.build_sam import build_sam2_video_predictor


def generate_masks(
		input_dir: list,
		output_dir: str,
		n_frames:int
):
	sam2_checkpoint = "segment_anything/checkpoints/sam2.1_hiera_large.pt"
	model_cfg = "//CT/VORF_GAN3/work/code/goae-inversion-olat/segment_anything/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
	predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
	inference_state = predictor.init_state(video_path=input_dir)
	predictor.reset_state(inference_state)

	ann_frame_idx = 0  # the frame index we interact with
	ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

	# Let's add a positive click at (x, y) = (210, 350) to get started
	points = np.array([[256, 256], [404, 82], [120, 42], [60, 256], [450, 256]], dtype=np.float32)
	# for labels, `1` means positive click and `0` means negative click
	labels = np.array([1, 1, 1, 0, 0], np.int32)

	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Propagate Masks
	# run propagation throughout the video and collect the results in a dict
	video_segments = {}  # video_segments contains the per-frame segmentation results
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# render the segmentation results every few frames
	generated_masks = []
	vis_frame_stride = 1
	for out_frame_idx in range(0, n_frames, vis_frame_stride):
		for out_obj_id, out_mask in video_segments[out_frame_idx].items():
			mask = out_mask.transpose(1, 2, 0).astype(np.uint8) * 255
			cv2.imwrite(os.path.join(output_dir, f'{out_frame_idx:03d}.png'), mask)
			generated_masks.append(mask/255)
	
	return generated_masks


