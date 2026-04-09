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
import cv2
from natsort import natsorted
import random
# import OpenEXR
# import Imath

from dataset_preprocessing.ffhq.download_ffhq import print_statistics
# from image_utils.data_util import load_exr
from image_utils.tonemap import apply_tonemap as process_image

try:
	import pyspng
except ImportError:
	pyspng = None

# ----------------------------------------------------------------------------
# Dataset paths

DATASET_PATHS = {
	'mpi': {
		'relight_roots': [
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-0',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-1',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-2',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor-0',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor-1',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor-2',
		],
		'debug_relight_roots': [
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor',
			'/CT/VORF_GAN5/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor',
		],
		'preprocess_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'labels_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'masks_root': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/10001/',
		'olat_root': '/CT/VORF_GAN5/static00/datasets/FOLAT_c2_align/',
		'light_dirs_path': '/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/LSX_light_positions_mpi.txt',
	},
	'merl': {
		'relight_roots': [
			'/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-full/',
			'/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor-full/',
		],
		'debug_relight_roots': [
			'/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/indoor-full/',
			'/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/relighting/outdoor-full/',
		],
		'preprocess_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'labels_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'masks_root': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/10001/',
		'olat_root': '/HPS/FacialRelighting/nobackup/data/FOLAT_c2/',
		'light_dirs_path': '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy',
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
	def seg_channels(self):
		return self.seg_shapes
	
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

# ----------------------------------------------------------------------------
import sys
import pdb
from typing import Any
class mpdb(pdb.Pdb):
	"""debugger for forked programs."""
	
	def interaction(self, *args: Any, **kwargs: Any) -> None:
		_stdin = sys.stdin
		try:
			sys.stdin = open("/dev/stdin")
			pdb.Pdb.interaction(self, *args, **kwargs)
		finally:
			sys.stdin = _stdin


def set_trace(*args: Any, **kwargs: Any) -> None:
	mpdb().set_trace(*args, **kwargs)


class LightstageImageFolderDataset(Dataset):
	def __init__(self,
				 path,  # Path to directory or zip.
				 resolution=None,  # Ensure specific resolution, None = highest available.
				 non_rebalance=True,
	             relight = False,
				 use_tonemap=False,
	             mask = True,
	             ref_interval=0,
				 eval_mode=False,
	             debug_mode=False,
	             dataset_name='mpi',
				 num_ids=10,
	             lightstage_res = 'half',
	             num_training_views=1,
	             **super_kwargs,  # Additional arguments for the Dataset base class.
				 ):

		self.eval_mode        = eval_mode
		self.debug_mode       = debug_mode
		self.dataset_name     = dataset_name
		dataset_paths         = DATASET_PATHS[self.dataset_name]
		# Paths
		if self.eval_mode:
			self._path        = path
			self.ignored_subjects = []
		else:
			self._path = dataset_paths['relight_roots']
			self.ignored_subjects = [
				'ID00000', 'ID00001', 'ID00022', 'ID00243', 'ID00277', 'ID00293', 'ID00307', 'ID00002', 'ID00004',
				'ID00023', 'ID00035', 'ID00058', 'ID00060', 'ID00063', 'ID00068', 'ID00088', 'ID00118', 'ID00119',
				'ID00248', 'ID00326', 'ID00353',
				'ID20068',
				'ID30032', 'ID30069'
				'ID40008', 'ID40011', 'ID40026', 'ID40027', 'ID40028', 'ID40029', 'ID40030', 'ID40031', 'ID40032',
				'ID40033', 'ID40034', 'ID40035', 'ID40036', 'ID40037', 'ID40038', 'ID40047', 'ID40050', 'ID40051',
				'ID40052', 'ID40053', 'ID40054', 'ID40055', 'ID40056', 'ID40067', 'ID40068', 'ID40088', 'ID40114',
				'ID20010',
				'ID20029', 'ID20031', 'ID20035', 'ID20037', 'ID20045', 'ID20059', 'ID20090', 'ID20113','ID20104',  # Evaluation Subjects
			]
			self.num_ids = num_ids
		
		self._landmarks_path = dataset_paths['preprocess_root']
		self._labels_path    = dataset_paths['labels_root']
		self._masks_path     = dataset_paths['masks_root']
		self._olat_path      = dataset_paths['olat_root']
		# Cameras
		if self.eval_mode:
			self.cameras          = ['Cam07']
			self.ref_camera       = 'Cam07' # single reference camera
		else:
			# self.cameras          = ['Cam06', 'Cam07', 'Cam15', 'Cam08', 'Cam10', 'Cam05']
			# self.cameras        = ['Cam06', 'Cam07']
			if self.dataset_name == 'mpi':
				if num_training_views == 1:
					self.cameras      = ['Cam06']
				elif num_training_views == 2:
					self.cameras      = ['Cam06', 'Cam39']
				elif num_training_views == 3:
					self.cameras      = ['Cam06', 'Cam39', 'Cam07']
				elif num_training_views >= 4:
					self.cameras      = ['Cam04', 'Cam06', 'Cam07', 'Cam23', 'Cam39']
				else:
					raise ValueError('Invalid number of views')
				
				self.ref_camera       = 'Cam04' # single reference camera
			else:
				self.cameras = ['Cam06', 'Cam07', 'Cam15']
				self.ref_camera       = 'Cam07' # single reference camera
			if self.num_ids > 100:
				self.num_emaps = 20
			else:
				self.num_emaps = 25
		# WARNING: Just for Debugging
		# FIXME: add changes to non debug mode
		if self.debug_mode:
			self._landmarks_path = dataset_paths['preprocess_root']
			self._labels_path    = dataset_paths['labels_root']
			self._masks_path     = dataset_paths['masks_root']
			self._olat_path      = dataset_paths['olat_root']
			self._path           = dataset_paths['debug_relight_roots']

			if self.num_ids > 100:
				self.num_emaps    = 20
			else:
				self.num_emaps    = 25
			# self.cameras      = ['Cam06']
			self.cameras      = ['Cam06', 'Cam23']
			# self.cameras      = ['Cam04', 'Cam06', 'Cam07', 'Cam23', 'Cam39']
			# self.cameras      = ['Cam06', 'Cam07', 'Cam15', 'Cam08', 'Cam10', 'Cam05']


		# Flags
		self._zipfile         = None
		self._relight         = relight
		self._use_tonemap     = use_tonemap
		self._mask            = mask
		# self._use_ref         = True if ref_interval > 0 else False
		
		# Tonemap operator
		if self.dataset_name == 'mpi':
			self.tonemap_olat = lambda x: pow(x, 0.5)
			# self.tonemap_olat = lambda x: np.clip(pow(np.clip(x, 0, 5), 0.5) * 1.5, 0, 1)
		else:
			# tonemap_olat = lambda x: (pow(x / (pow(2, 16)), 0.5) * 255)
			self.tonemap_olat = lambda x: pow(x, 0.5)
			# tonemap_olat = lambda x: 1/(x+1)
		
		# Data
		if type(self._path) is list:
			for path in self._path:
				assert os.path.isdir(path), f'{path} not found!'
			self._type = 'dir'
			self._all_fnames, self._ref_image_fnames = self._get_all_filenames()
		elif os.path.isdir(self._path):
				self._type = 'dir'
				self._all_fnames, self._ref_image_fnames = self._get_all_filenames()
		elif self._file_ext(self._path) == '.zip':
			self._type = 'zip'
			self._all_fnames = set(self._get_zipfile().namelist())
		else:
			raise IOError('Path must point to a directory or zip')

		PIL.Image.init()
		self._image_fnames = self._all_fnames
		if len(self._image_fnames) == 0:
			raise IOError('No image files found in the specified path')
		name = os.path.splitext(os.path.basename(self._path[0]))[0]
		raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution)[0].shape)
		
		print(f'Image SHAPE {raw_shape} and other args {super_kwargs}')
		print(self.cameras)
		if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
			raise IOError('Image files do not match the specified resolution')

		#light directions
		self._lightdirs = self._get_lightdirs(dataset_name)
		self._n_olats = len(self._lightdirs)
		
		full_light_indices = [0, 20, 41, 62, 83, 104, 125, 146, 167, 188, 209, 230, 251, 272, 293, 314, 335, 348, 349]
		bad_light_indices = [270, 296,
		                     317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
		                     334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]
		
		# ignore_indices = sorted(set(full_light_indices + bad_light_indices))
		# self.all_light_indices = [x for x in range(self._n_olats) if (x not in ignore_indices and x < 300)]
		
		self.all_light_indices = [x for x in range(self._n_olats) if (x not in full_light_indices and x < 300)]
		self.all_light_indices = [x for x in self.all_light_indices if x not in bad_light_indices]
		
		# HACK: We are not sampling all the lights
		# FIXME: add all the lights for complete relighting
		# self.all_light_indices = self.all_light_indices[0:150]
		if lightstage_res == 'half':
			freq = 2
		elif lightstage_res == 'full':
			freq = 1
		else:
			raise NotImplementedError
		self.all_light_indices = [x for idx, x in enumerate(self.all_light_indices) if idx%freq==0] # sample every 2nd OLAT

		print( 'Total Lights: ', len(self.all_light_indices), '\n' , self.all_light_indices, '\n')
		
		super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
	
	@staticmethod
	def _file_ext(fname):
		return os.path.splitext(fname)[1].lower()
	
	def _get_all_filenames(self):
		"""
		Given the root directory, returns a list of tuples of image paths and corresponding mask paths.

		Parameters:
		- root_dir (str): The root directory containing the dataset.
		- cameras (list): The camera name to be sampled.
		- ignored_subjects (list): List of subject names to be ignored.

		Returns:
		- List where:
			- element is the image path
		"""
		images = []
		ref_images = []
		train_id_count = 0

		dir_name = 'crop' if self.eval_mode else 'images'
		if self.eval_mode:
			# Iterate over all the directories inside root_dirs
			for path_idx, path in enumerate(self._path):
				for root, _, files in os.walk(path):
					# Ensure we're not inside an ignored subject's directory
					if not any(subject in root for subject in self.ignored_subjects):
						# Ensure we're in a directory that contains both images and masks
						if 'crop' in root:
							for file in files:
								# Check if the file is from the desired camera
								for camera in self.cameras:
									cam, id, emap = os.path.splitext(file)[0].split('_')
									if (camera in file and os.path.isfile(os.path.join(self._landmarks_path, id, 'transform', file[:13]+'.txt'))):
										# Get the image path
										image_path = os.path.join(root, file)
										images.append(image_path)

			return images, images
		else:
			for path in self._path:
				for sub_name in natsorted(glob.glob1(path, '*')):
					# Ensure we're not inside an ignored subject's directory
					num_images = 0
					if not any(name in sub_name for name in self.ignored_subjects):
						if train_id_count < self.num_ids:
							train_id_count += 1
						else:
							break

						root = os.path.join(path, sub_name)
						# Ensure we're in a directory that contains both images and masks
						if os.path.exists(os.path.join(root, 'images')):
							for camera in self.cameras:
								files = natsorted(glob.glob(os.path.join(root, 'images', f'{camera}*.png')))
								if self.debug_mode:
									if self.num_ids > 1:
										random.shuffle(files)
								files = files[:self.num_emaps]
								landmarks_file = os.path.join(self._landmarks_path, sub_name, 'transform',f'{camera}_{sub_name}.txt')
								if os.path.isfile(landmarks_file):
									images.extend(files)
									num_images += len(files)
						
						print(f'{sub_name} has training {num_images} images')
						
				print(f'Using a total of {train_id_count} training subjects')

			return images, ref_images

	def _get_lightdirs(self, dataset_name):
		light_dirs_path = DATASET_PATHS[dataset_name]['light_dirs_path']
		if dataset_name == 'mpi':
			light_dirs = np.loadtxt(light_dirs_path).astype(np.float32)
		else:
			light_dirs = np.load(light_dirs_path).astype(np.float32)
		return light_dirs / np.linalg.norm(light_dirs, axis=1)[..., None]

	def _get_zipfile(self):
		assert self._type == 'zip'
		if self._zipfile is None:
			self._zipfile = zipfile.ZipFile(self._path)
		return self._zipfile
	
	def _open_file(self, fname):
		if self._type == 'dir':
			return open(os.path.join(fname), 'rb')
		if self._type == 'zip':
			return self._get_zipfile().open(fname, 'r')
		return None
	
	def _open_segfile(self, fname):
		if self._type == 'dir':
			return open(os.path.join(self._seg_path, fname), 'rb')
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
		fname = self._image_fnames[raw_idx]
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		if self.eval_mode:
			image = (np.float32(cv2.imread(fname, -1)[:, :, ::-1]))
			image = (image).transpose(2, 0, 1)  # HWC => CHW
		else:
			scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
			with self._open_file(fname) as f:
				# MPI-LS
				if self.dataset_name == 'mpi':
					image = (np.float32(cv2.imread(fname, -1)[:, :, ::-1]))
				else:
					image = (np.float32(cv2.rotate(cv2.imread(fname, -1)[:, :, ::-1], cv2.ROTATE_90_CLOCKWISE)))
				image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1) #HWC => CHW
		
		return image, fname
	
	def _load_raw_ref_image(self, raw_idx, resolution=None):
		fname = self._ref_image_fnames[raw_idx]
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
		with self._open_file(fname) as f:
			image = (np.float32(cv2.rotate(cv2.imread(fname, -1)[:, :, ::-1], cv2.ROTATE_90_CLOCKWISE)))
			image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1)  # HWC => CHW
		
		return image, fname

	def _load_raw_olat_image(self, fname, olat_idx, resolution=None):
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
		# MPI-LS
		if self.dataset_name=='mpi':
			olat_fname = os.path.join(self._olat_path, cam, id, f'{olat_idx:03d}.exr')
			# image = load_exr(olat_fname)
			image = (np.float32(cv2.imread(olat_fname, -1)[:, :, ::-1]))
			# Color correction for OLAT the images
			if self._use_tonemap:
				image = process_image(image)
				image = (image/65535)*255
		else:
			olat_fname = os.path.join(self._olat_path, cam, id, f'{olat_idx:03d}.png')
			with self._open_file(olat_fname) as f:
				image = (np.float32(cv2.imread(olat_fname, -1)[:, :, ::-1]))
			# Color correction for OLAT the images
			if self._use_tonemap:
				image = self.tonemap_olat(image)

		# Crop and Align
		image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1)  # HWC => CHW
		return image

	def _load_raw_olat_ref_image(self, fname, olat_idx, resolution=None):
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
		olat_fname = os.path.join(self._olat_path, cam, id, f'{olat_idx:03d}.png' )
		with self._open_file(olat_fname) as f:
			image = (np.float32(cv2.imread(olat_fname, -1)[:, :, ::-1]))
			image = self._resize_n_crop_numpy(image, scale_crop_params).transpose(2, 0, 1)  # HWC => CHW
			if self._use_tonemap:
				image = self.tonemap_olat(image)

		return image

	def _load_mask(self, fname):
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		if 0 <= int(id[3:]) < 106:
			emap = 'EMAP-350'
		elif 106 <= int(id[3:]) < 211:
			emap = 'EMAP-418'
		elif 211 <= int(id[3:]) < 313:
			emap = 'EMAP-486'
		else:
			emap = 'EMAP-999'
			
		if self.dataset_name=='mpi':
			# MPI-LS
			mask_fname = os.path.join(self._masks_path, id, 'sam', f'{cam}_{id}.png')
			scale_crop_params = np.loadtxt(os.path.join(self._landmarks_path, id, 'transform', f'{cam}_{id}.txt'))
			image = cv2.imread(mask_fname)
			image = self._resize_n_crop_numpy(image, scale_crop_params)
		else:
			mask_fname = os.path.join(self._masks_path, id, 'mask_seg', f'{cam}_{id}_{emap}.png')
			# mask_eye = os.path.join(self._masks_path, id, 'mask', f'{cam}_{id}_{emap}.png')
			image = cv2.imread(mask_fname) # * cv2.imread(mask_eye)[:, :, 0:1]

		return image[:,:,0:1].transpose(2, 0, 1)

	def _load_ref_mask(self, fname):
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		if 0 <= int(id[3:]) < 106:
			emap = 'EMAP-350'
		elif 106 <= int(id[3:]) < 211:
			emap = 'EMAP-418'
		elif 211 <= int(id[3:]) < 313:
			emap = 'EMAP-486'
		else:
			emap = 'EMAP-999'

		mask_fname = os.path.join(self._labels_path, id, 'mask_seg', f'{cam}_{id}_{emap}.png')
		with self._open_file(mask_fname) as f:
			image = cv2.imread(mask_fname)[:, :, 0:1]
		return image.transpose(2, 0, 1)

	def _load_eval_mask(self, fname):
		cam, id, emap = os.path.splitext(os.path.basename(fname))[0].split('_')
		if 0 <= int(id[3:]) < 106:
			emap = 'EMAP-350'
		elif 106 <= int(id[3:]) < 211:
			emap = 'EMAP-418'
		elif 300 <= int(id[3:]) < 360:
			emap = 'EMAP-860'
		else:
			emap = 'EMAP-999'

		mask_fname = os.path.join(self._masks_path, id, 'mask_seg', f'{cam}_{id}_{emap}.png')
		# mask_eye = os.path.join(self._masks_path, id, 'mask', f'{cam}_{id}_{emap}.png')
		image = cv2.imread(mask_fname)[:, :, 0:1] # * cv2.imread(mask_eye)[:, :, 0:1]

		return image.transpose(2, 0, 1)
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
	
	
	def _load_invalid_list(self):
		fname = 'invalids.txt'
		with open(os.path.join(self._path, fname), 'r') as f:
			lines = f.readlines()
		lines = [line.split('\n')[0].replace('\\', '/') for line in lines]
		return lines
	
	def _load_raw_labels(self):
		# Iterate over all the directories inside root_dir
		labels = []
		ref_labels = []
		train_id_count = 0
		if self.eval_mode:
			for path in self._path:
				for root, _, files in os.walk(path):
					# Ensure we're not inside an ignored subject's directory
					if not any(subject in root for subject in self.ignored_subjects):
						# Ensure we're in a directory that contains both images and masks
						if 'crop' in root:
							for file in files:
								# Check if the file is from the desired camera
								for camera in self.cameras:
									if camera in file and file.endswith('.png'):
										id = os.path.basename(os.path.dirname(root))
										c_path = os.path.join(self._labels_path, id, 'camera', 'dataset_dict.json')
										with open(c_path, 'r') as c_f:
											labels_dict = json.load(c_f)
										# for camera in self.cameras:
										for key in labels_dict:
											if camera in key:
												labels.append(labels_dict[key])
			labels = np.array(labels).astype(np.float32)
			return labels, labels
		else:
			for path in self._path:
				for sub_name in natsorted(glob.glob1(path, '*')):
					num_cams=0
					# Ensure we're not inside an ignored subject's directory
					if not any(name in sub_name for name in self.ignored_subjects):
						if train_id_count < self.num_ids:
							train_id_count += 1
						else:
							break

						root = os.path.join(path, sub_name)
						# set_trace()
						# Ensure we're in a directory that contains both images and masks
						if os.path.exists(os.path.join(root, 'images')):
							for camera in self.cameras:
								files = natsorted(glob.glob1(os.path.join(root, 'images'), f'{camera}*.png'))
								# if self.debug_mode:
								files = files[:self.num_emaps]
								for file in files:
									if camera in file and file.endswith('.png'):
										# id = os.path.basename(os.path.dirname(root))
										assert sub_name in root, print(f'{root} folder does not contain {sub_name}')
										try:
											c_path = os.path.join(self._labels_path, sub_name, 'camera', 'dataset_dict.json')
											with open(c_path, 'r') as c_f:
												labels_dict = json.load(c_f)
										except FileNotFoundError:
											c_path = os.path.join(self._labels_path, sub_name, 'camera', 'dataset.json')
											with open(c_path, 'r') as c_f:
												raw_labels_dict = json.load(c_f)

											# Convert to dataset_dict.json format
											labels_dict = {}
											for entry in raw_labels_dict['labels']:
												labels_dict[entry[0]]=entry[1]

										# for camera in self.cameras:
										for key in labels_dict:
											if camera in key:
												# set_trace()
												ref_key = key.replace(camera, self.ref_camera)
												# if labels_dict.get(ref_key) is not None:
												num_cams += 1
												labels.append(labels_dict[key])
												# ref_labels.append(labels_dict[ref_key])
						
						# print(f'{sub_name} has training {num_cams} cameras')


			labels = np.array(labels).astype(np.float32)
			# ref_labels = np.array(ref_labels).astype(np.float32)

			return labels, ref_labels
	
	def _get_raw_labels(self):
		if self._raw_labels is None:
			self._raw_labels, self._raw_ref_labels = self._load_raw_labels() if self._use_labels else None
			if self._raw_labels is None:
				self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
			assert isinstance(self._raw_labels, np.ndarray)
			assert self._raw_labels.shape[0] == self._raw_shape[0], print(f'{self._raw_labels.shape[0]} Cams != {self._raw_shape[0]} Images ')
			assert self._raw_labels.dtype in [np.float32, np.int64]
			if self._raw_labels.dtype == np.int64:
				assert self._raw_labels.ndim == 1
				assert np.all(self._raw_labels >= 0)
			# Reference Labels
			# if self._raw_ref_labels is None:
			# 	self._raw_ref_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
			# assert isinstance(self._raw_ref_labels, np.ndarray)
			# assert self._raw_ref_labels.shape[0] == self._raw_shape[0]
			# assert self._raw_ref_labels.dtype in [np.float32, np.int64]
			# if self._raw_ref_labels.dtype == np.int64:
			# 	assert self._raw_ref_labels.ndim == 1
			# 	assert np.all(self._raw_ref_labels >= 0)
		
		# return self._raw_labels, self._raw_ref_labels
		return self._raw_labels, self._raw_labels
	
	def get_label(self, idx):
		label = self._get_raw_labels()[0][self._raw_idx[idx]]
		# if label.dtype == np.int64:
		#     onehot = np.zeros(self.label_shape, dtype=np.float32)
		#     onehot[label] = 1
		#     label = onehot
		return label.copy()
	
	def get_ref_label(self, idx):
		label = self._get_raw_labels()[1][self._raw_idx[idx]]
		return label.copy()


#----------------------------------------------------------------------------


class LightstageCameraLabeledDataset(LightstageImageFolderDataset):
	
	def __getitem__(self, idx):
		image, fname = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
		label = self.get_label(idx)
		assert isinstance(image, np.ndarray)
		assert list(image.shape) == self.image_shape, print(f'Instead found {fname} image of type {image.dtype}')
		assert image.dtype == np.float32

		if self.eval_mode:
			mask = self._load_eval_mask(fname)
			image = np.concatenate((image, mask), axis=0)

		if self._relight:
			if self.dataset_name == 'mpi':
				# WARNING: using subset of OLATs to see if it works
				olat_idx = random.choice(self.all_light_indices)
			else:
				olat_idx = random.randrange(self._n_olats)
			# if self.debug_mode:
			# 	olat_idx = 82

			dirs = self._lightdirs[olat_idx]
			olat_image = self._load_raw_olat_image(fname, olat_idx)
			# print(fname, self._use_tonemap, olat_idx, dirs, olat_image.max(), olat_image.min())
			if (cv2.imread(fname).shape) == None:
				print(f'The faulty image is : {idx}:{fname}')
				# print(f'{idx}:{fname} & shape = {cv2.imread(fname).shape}')
			# else:
			# 	print(f'The faulty image is : {idx}:{fname}')
			
			mask = self._load_mask(fname)

			image = np.concatenate((image, mask), axis=0)
			olat_image = olat_image
			return image, label, olat_image, dirs
		else:
			if self.eval_mode:
				return image.copy(), label
			else:
				# return image.copy(), label, ref_image.copy(), ref_label
				return image.copy(), label
