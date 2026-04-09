# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import glob
from natsort import natsorted
import collections
import cv2

from inversion.utils.debugger import set_trace

try:
	import pyspng
except ImportError:
	pyspng = None

#----------------------------------------------------------------------------
# Dataset paths

DATASET_PATHS = {
	'mpi': {
		'preprocess_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'labels_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'masks_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'eval_relight_root': '/CT/VORF_GAN5/static00/datasets/evaluation/indoor-test/',
	},
	'merl': {
		'preprocess_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'labels_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'masks_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'eval_relight_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-test/',
	},
}

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
	def __init__(self,
	             name,                   # Name of the dataset.
	             raw_shape,              # Shape of the raw image data (NCHW).
	             max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
	             use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
	             xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
	             random_seed = 0,        # Random seed to use when applying max_size.
	             ):
		self._name = name
		self._raw_shape = list(raw_shape)
		self._use_labels = use_labels
		self._raw_labels = None
		self._label_shape = None
		self._latent_2d_all = None

		# Apply max_size.
		self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
		if (max_size is not None) and (self._raw_idx.size > max_size):
			np.random.RandomState(random_seed).shuffle(self._raw_idx)
			self._raw_idx = np.sort(self._raw_idx[:max_size])

		# Apply xflip.
		self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
		if xflip:
			self._raw_idx = np.tile(self._raw_idx, 2)
			self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

	def _get_raw_labels(self):
		if self._raw_labels is None:
			self._raw_labels = self._load_raw_labels() if self._use_labels else None
			if self._raw_labels is None:
				self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
			assert isinstance(self._raw_labels, np.ndarray)
			assert self._raw_labels.shape[0] == self._raw_shape[0]
			assert self._raw_labels.dtype in [np.float32, np.int64]
			if self._raw_labels.dtype == np.int64:
				assert self._raw_labels.ndim == 1
				assert np.all(self._raw_labels >= 0)
		return self._raw_labels

	def close(self): # to be overridden by subclass
		pass

	def _load_raw_image(self, raw_idx): # to be overridden by subclass
		raise NotImplementedError

	def _load_raw_labels(self): # to be overridden by subclass
		raise NotImplementedError

	def __getstate__(self):
		return dict(self.__dict__, _raw_labels=None)

	def __del__(self):
		try:
			self.close()
		except:
			pass

	def __len__(self):
		return self._raw_idx.size

	def __getitem__(self, idx):
		image = self._load_raw_image(self._raw_idx[idx])
		assert isinstance(image, np.ndarray)
		assert list(image.shape) == self.image_shape
		assert image.dtype == np.uint8
		if self._xflip[idx]:
			assert image.ndim == 3 # CHW
			image = image[:, :, ::-1]
		return image.copy(), self.get_label(idx)

	def get_label(self, idx):
		label = self._get_raw_labels()[self._raw_idx[idx]]
		if label.dtype == np.int64:
			onehot = np.zeros(self.label_shape, dtype=np.float32)
			onehot[label] = 1
			label = onehot
		return label.copy()

	def get_details(self, idx):
		d = dnnlib.EasyDict()
		d.raw_idx = int(self._raw_idx[idx])
		d.xflip = (int(self._xflip[idx]) != 0)
		d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
		return d

	@property
	def name(self):
		return self._name

	@property
	def image_shape(self):
		return list(self._raw_shape[1:])

	@property
	def num_channels(self):
		assert len(self.image_shape) == 3 # CHW
		return self.image_shape[0]

	@property
	def resolution(self):
		assert len(self.image_shape) == 3 # CHW
		assert self.image_shape[1] == self.image_shape[2]
		return self.image_shape[1]

	@property
	def label_shape(self):
		if self._label_shape is None:
			raw_labels = self._get_raw_labels()
			if raw_labels.dtype == np.int64:
				self._label_shape = [int(np.max(raw_labels)) + 1]
			else:
				self._label_shape = raw_labels.shape[1:]
		return list(self._label_shape)

	@property
	def label_dim(self):
		assert len(self.label_shape) == 1
		return self.label_shape[0]

	@property
	def has_labels(self):
		return any(x != 0 for x in self.label_shape)

	@property
	def has_onehot_labels(self):
		return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
	def __init__(self,
	             path,                   # Path to directory or zip.
	             resolution      = None, # Ensure specific resolution, None = highest available.
	             use_512         = False,
	             use_mask         = False,
	             non_rebalance   = True,
	             **super_kwargs,         # Additional arguments for the Dataset base class.
	             ):
		self._path = path
		self._zipfile = None
		self._resolution = resolution
		self.use_512 = use_512
		self.cameras = ['Cam06', 'Cam07']
		self.use_mask = use_mask

		if os.path.isdir(self._path):
			self._type = 'dir'
			# self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=os.path.join(self._path, 'crop')) for root, _dirs, files in os.walk(os.path.join(self._path, 'crop')) for fname in files}
			self._all_fnames = set()
			#WARNING: Configure Cameras for all views
			for root, _dirs, files in os.walk(os.path.join(self._path, 'crop')):
				for fname in files:
					if fname[0:5] in self.cameras:
						self._all_fnames.add(os.path.relpath(os.path.join(root, fname), start=os.path.join(self._path, 'crop')))

		elif self._file_ext(self._path) == '.zip':
			self._type = 'zip'
			self._all_fnames = set(self._get_zipfile().namelist())
		else:
			raise IOError('Path must point to a directory or zip')

		PIL.Image.init()
		self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
		if non_rebalance:
			self._image_fnames = [fname for fname in self._image_fnames ]
		if len(self._image_fnames) == 0:
			raise IOError('No image files found in the specified path')
		name = os.path.splitext(os.path.basename(self._path))[0]
		if self.use_512:
			raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution)[0].shape)
		else:
			raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution).shape)
		if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
			raise IOError('Image files do not match the specified resolution')
		super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

	@staticmethod
	def _file_ext(fname):
		return os.path.splitext(fname)[1].lower()

	def _get_zipfile(self):
		assert self._type == 'zip'
		if self._zipfile is None:
			self._zipfile = zipfile.ZipFile(self._path)
		return self._zipfile

	def _open_file(self, fname):
		if self._type == 'dir':
			if (os.path.isfile(os.path.join(self._path, fname))) is True:
				return open(os.path.join(self._path, fname), 'rb')
			else:
				try:
					return open(os.path.join(self._path, 'crop', fname), 'rb')
				except FileNotFoundError:
					return open(os.path.join(self._path, 'camera', fname), 'rb')


		if self._type == 'zip':
			return self._get_zipfile().open(fname, 'r')
		return None

	def _open_mask_file(self, fname):
		if self._type == 'dir':
			try:
				# return open(os.path.join(self._path, 'mask_seg', fname), 'rb')
				return open(os.path.join(self._path, 'mask_rmbg2', fname), 'rb')
			except FileNotFoundError:
				print("Mask Path Missing")
				return None

		return None

	def close(self):
		try:
			if self._zipfile is not None:
				self._zipfile.close()
		finally:
			self._zipfile = None

	def __getstate__(self):
		return dict(super().__getstate__(), _zipfile=None)

	def _load_raw_image(self, raw_idx, resolution=None):
		fname = self._image_fnames[raw_idx]
		if self.use_512:
			with self._open_file(fname) as f:

				image = PIL.Image.open(f)
				image512 = image
				if resolution:
					image = image.resize((resolution, resolution))
				image = np.array(image)
				image512 = np.array(image512)
			if image.ndim == 2:
				image = image[:, :, np.newaxis] # HW => HWC
				image512 = image512[:, :, np.newaxis]

			if self.use_mask:
				mask512 = cv2.imread(os.path.join(self._path, 'mask_rmbg2', fname))
				mask512 = cv2.cvtColor(mask512, cv2.COLOR_BGR2RGB) / 255
				image512 = image512 * mask512
				image = cv2.resize(image512, (resolution, resolution))
				
				
				# import pdb; pdb.set_trace()
				
				#
				#
				# with self._open_mask_file(fname) as f:
				# 	mask = PIL.Image.open(f)
				# 	mask512 = mask
				# 	if resolution:
				# 		mask = mask.resize((resolution, resolution))
				# 	mask = np.array(mask) // 255
				# 	mask512 = np.array(mask512) // 255
				#
				# 	image *= mask
				# 	image512 *= mask512

			image = image.transpose(2, 0, 1) # HWC => CHW
			image512 = image512.transpose(2, 0, 1) # HWC => CHW
			return image, image512
		else:
			with self._open_file(fname) as f:
				image = PIL.Image.open(f)
				if resolution:
					image = image.resize((resolution, resolution))
				image = np.array(image)
			if image.ndim == 2:
				image = image[:, :, np.newaxis] # HW => HWC
			image = image.transpose(2, 0, 1) # HWC => CHW
			return image




	def _load_raw_labels(self):
		fname = 'dataset.json'
		# fname = 'label.json'
		# if fname not in self._all_fnames:
		#     return None
		with self._open_file(fname) as f:
			labels = json.load(f)['labels']
		if labels is None:
			return None
		labels = dict(labels)
		labels = [labels[fname.replace('\\', '/').replace('png','jpg')] for fname in self._image_fnames]
		labels = np.array(labels)
		labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
		return labels




class CameraLabeledDataset(ImageFolderDataset):

	def __getitem__(self, idx):
		# image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		if self.use_512:
			image, image512 = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		else:
			image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		label = self.get_label(idx)
		# latent_2d = self.get_latent_2d(idx)
		assert isinstance(image, np.ndarray)
		assert list(image.shape) == self.image_shape
		# assert image.dtype == np.uint8, f'image is of type {image.dtype}'
		if self._xflip[idx]:
			assert image.ndim == 3 # CHW
			image = image[:, :, ::-1]
			if self._use_labels:
				assert label.shape == (25,)
				label[[1,2,3,4,8]] *= -1

		if self.use_512:
			return image.copy(), label, image512.copy(), self._image_fnames[idx]
		else:
			return image.copy(), label, self._image_fnames[idx]


class EvalImageFolderDataset(Dataset):
	def __init__(self,
	             scan_name,                   # Path to directory or zip.
	             resolution      = None, # Ensure specific resolution, None = highest available.
	             use_512         = False,
	             non_rebalance   = True,
	             input_cam       = 'Cam07',
	             num_emaps       = 5,
	             dataset_name    = None,
	             cam_list        = None,
	             **super_kwargs,         # Additional arguments for the Dataset base class.
	             ):
		self._scan_name = scan_name
		self._zipfile = None
		self._type = 'dir'
		self._resolution = resolution
		self.use_512 = use_512
		self.input_cam = input_cam
		self.num_emaps = num_emaps
		self.dataset_name = dataset_name
		self.render_camera_names = cam_list
		dataset_paths = DATASET_PATHS[dataset_name]
		
		if dataset_name == 'merl':
			self.eval_path       = dataset_paths['eval_relight_root']
			self._landmarks_path = dataset_paths['preprocess_root']
			self._labels_path    = dataset_paths['labels_root']
			self._masks_path     = dataset_paths['masks_root']
			self._path           = os.path.join(dataset_paths['preprocess_root'], scan_name)
		elif dataset_name == 'mpi':
			self.eval_path       = dataset_paths['eval_relight_root']
			self._landmarks_path = dataset_paths['preprocess_root']
			self._labels_path    = dataset_paths['labels_root']
			self._masks_path     = dataset_paths['masks_root']
			self._path           = os.path.join(dataset_paths['preprocess_root'], scan_name)
			# self.render_camera_names = ['Cam04', 'Cam06', 'Cam07', 'Cam23', 'Cam39']
			# self.render_camera_names = cam_list
		
		else:
			raise NotImplementedError

		self._all_cam_names = natsorted(glob.glob1(os.path.join(self._path, 'crop'), '*.png'))

		self._input_imgs = []
		all_fnames = natsorted(glob.glob1(os.path.join(self.eval_path, self._scan_name, 'images'), '*.png'))
		for img_name in all_fnames:
			if self.input_cam in img_name:
				self._input_imgs.append(img_name)

		PIL.Image.init()
		# Number of different environment maps as input
		self._image_fnames = self._input_imgs[:self.num_emaps]

		print(f"Inverting following images: {self._image_fnames}")

		# if non_rebalance:
		#     self._image_fnames = [fname for fname in self._image_fnames ]
		if len(self._image_fnames) == 0:
			raise IOError(f'No image files found in the specified path in {self.eval_path}')
		# name = os.path.splitext(os.path.basename(self._path))[0]
		name = scan_name
		if self.use_512:
			raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution)[0].shape)
		else:
			raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution).shape)
		if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
			raise IOError('Image files do not match the specified resolution')
		super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

	@staticmethod
	def _file_ext(fname):
		return os.path.splitext(fname)[1].lower()

	def _resize_n_crop_numpy(self, img, scale_crop_param, target_size=1024, output_size=512, mask=None):

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

	def _get_zipfile(self):
		assert self._type == 'zip'
		if self._zipfile is None:
			self._zipfile = zipfile.ZipFile(self._path)
		return self._zipfile

	def _open_file(self, fname):
		if self._type == 'dir':
			if (os.path.isfile(os.path.join(self._path, fname))) is True:
				return open(os.path.join(self._path, fname), 'rb')
			else:
				try:
					return open(os.path.join(self._path, 'crop', fname), 'rb')
				except FileNotFoundError:
					try:
						return open(os.path.join(self._path, 'camera', fname), 'rb')
					except FileNotFoundError:
						return open(os.path.join(self.eval_path, self._scan_name, 'images', fname), 'rb')



		if self._type == 'zip':
			return self._get_zipfile().open(fname, 'r')
		return None

	def close(self):
		try:
			if self._zipfile is not None:
				self._zipfile.close()
		finally:
			self._zipfile = None

	def __getstate__(self):
		return dict(super().__getstate__(), _zipfile=None)

	def _load_raw_image(self, raw_idx, resolution=None):
		assert self.use_512 is True
		fname = self._image_fnames[raw_idx]

		cam, id, _ = fname.split('_')
		image = cv2.imread(os.path.join(self.eval_path, self._scan_name, 'images', fname))[:, :, ::-1]
		scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
		H, W, _ = image.shape
		if H == 1030:  # image needs to be rotated to portrait mode
			image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
		image = self._resize_n_crop_numpy(image, scale_crop_params)

		# Make Copies
		image512 = np.array(image).transpose(2, 0, 1)
		image = cv2.resize(image, (resolution, resolution)).transpose(2, 0, 1)
		
		if self.dataset_name == 'mpi':
			try:
				mask_path = os.path.join(self._masks_path, self._scan_name, 'mask_rmbg2', f'{cam}_{id}.png')
				assert os.path.isfile(mask_path), f'{mask_path} does not exist'
			except AssertionError:
				mask_path = os.path.join(self._masks_path, self._scan_name, 'bgMatting', f'{cam}_{id}.png')
				assert os.path.isfile(mask_path), f'{mask_path} does not exist'
		elif self.dataset_name == 'merl':
			mask_path = os.path.join(self._masks_path, self._scan_name, 'mask_rmbg2', fname)
		else:
			raise NotImplementedError
		
		if os.path.isfile(mask_path):
			mask512 = cv2.imread(mask_path)
			if (mask512.shape[0] == 4096) or (mask512.shape[0] == 1024):
				mask512 = self._resize_n_crop_numpy(mask512, scale_crop_params)
			mask = cv2.resize(mask512, (resolution, resolution))

			image512 = image512 * (np.array(mask512).transpose(2, 0, 1) // 255)
			image = image * (np.array(mask).transpose(2, 0, 1) // 255)


		return image, image512

		# # image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1)  # HWC => CHW
		#
		# image = np.float32(cv2.imread(fname)[:, :, ::-1])
		# H, W, _ = image.shape
		# if H == 1030:  # image needs to be rotated to portrait mode
		#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
		#
		# if self.use_512:
		#     with self._open_file(fname) as f:
		#         image = PIL.Image.open(f)
		#         image512 = image
		#         if resolution:
		#             image = image.resize((resolution, resolution))
		#         image = np.array(image)
		#         image512 = np.array(image512)
		#     if image.ndim == 2:
		#         image = image[:, :, np.newaxis] # HW => HWC
		#         image512 = image512[:, :, np.newaxis]
		#     image = image.transpose(2, 0, 1) # HWC => CHW
		#     image512 = image512.transpose(2, 0, 1) # HWC => CHW
		#     return image, image512
		# else:
		#     with self._open_file(fname) as f:
		#         image = PIL.Image.open(f)
		#         if resolution:
		#             image = image.resize((resolution, resolution))
		#         image = np.array(image)
		#     if image.ndim == 2:
		#         image = image[:, :, np.newaxis] # HW => HWC
		#     image = image.transpose(2, 0, 1) # HWC => CHW
		#     return image

	def _load_raw_labels(self):
		fname = 'dataset.json'
		with self._open_file(fname) as f:
			labels = json.load(f)['labels']
		if labels is None:
			return None
		labels = dict(labels)
		if self.dataset_name == 'mpi':
			render_labels = {}
			for key, val in labels.items():
				cam_name = key.split('_')[0]
				if cam_name in self.render_camera_names:
					render_labels[key] = val
		else:
			render_labels = labels
		
		return render_labels

	def _get_raw_labels(self):
		if self._raw_labels is None:
			self._raw_labels = self._load_raw_labels() if self._use_labels else None
		return self._raw_labels

	def _get_render_labels(self):
		ordered_labels = collections.OrderedDict(natsorted(self._raw_labels.items()))
		labels = [val for key, val in ordered_labels.items()]
		labels = np.array(labels)
		return labels

	def _get_input_label(self, idx):
		cam_name = self._image_fnames[idx].split('_')[0]
		label = [value for key, value in self._get_raw_labels().items() if cam_name in key]
		label = np.array(label)

		if label.dtype == np.int64:
			onehot = np.zeros(self.label_shape, dtype=np.float32)
			onehot[label] = 1
			label = onehot
		return label.copy()

	def _get_ref_images(self, idx):
		image_name = self._image_fnames[idx]
		emap = image_name.split('_')[-1]

		all_fnames = natsorted(glob.glob(os.path.join(self.eval_path, self._scan_name, 'images/') + f'*{emap}'))
		images = []
		image_names = []
		for fname in all_fnames:
			scan_name = os.path.basename(fname)
			cam ,id, _ = scan_name.split('_')
			try:
				scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
				image = np.float32(cv2.imread(fname)[:, :, ::-1])
				H, W, _ = image.shape
				if H == 1030:  # image needs to be rotated to portrait mode
					image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

				image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1)  # HWC => CHW
				images.append(image)
				image_names.append(fname)
			except FileNotFoundError:
				# print(f"Skipping {fname}")
				continue

		# image = (np.float32(cv2.rotate(cv2.imread(fname)[:, :, ::-1], cv2.ROTATE_90_CLOCKWISE)))
		return np.array(images), image_names


class EvalCameraLabeledDataset(EvalImageFolderDataset):
	def __getitem__(self, idx):
		# print(f'----------------')
		# print(f"Batch ID: {idx}")
		# print(f'----------------')
		if self.use_512:
			image, image512 = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		else:
			image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		# input camera pose
		label = self._get_input_label(idx)

		assert isinstance(image, np.ndarray)
		assert list(image.shape) == self.image_shape
		assert image.dtype == np.uint8

		# render cams
		render_labels = self._get_render_labels()
		if self.dataset_name == 'merl':
			assert len(render_labels) > 10
		elif self.dataset_name == 'mpi':
			assert len(render_labels) > 4, f'Found only {len(render_labels)} labels: {self.render_camera_names}'
		else:
			raise NotImplementedError

		# ground truth images
		reference_images, reference_image_names = self._get_ref_images(idx)
		print('=================================')
		print(len(reference_images), len(render_labels), self._image_fnames[idx])
		print('=================================')
		# assert len(reference_images) == len(render_labels), print(len(reference_images), len(render_labels), self._image_fnames[idx])

		# print(image.shape, image512.shape, reference_images.shape)

		if self.use_512:
			return image.copy(), label, image512.copy(), render_labels, reference_images, self._image_fnames[idx], reference_image_names
		else:
			return image.copy(), label, render_labels, reference_images, self._image_fnames[idx], reference_image_names


