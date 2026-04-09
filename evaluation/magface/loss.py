import cv2
import torch
import numpy as np
from torchvision import transforms
# Initialize Model
import sys
from evaluation.magface.models import iresnet
from collections import OrderedDict
from tqdm import tqdm
from termcolor import cprint
import os
import torch.nn.functional as F
import torch.nn as nn
import torch


def load_features(args):
	if args['arch'] == 'iresnet100':
		features = iresnet.iresnet34(
			pretrained=False,
			num_classes=args['embedding_size'],
		)
	# elif args.arch == 'iresnet18':
	# 	features = iresnet.iresnet18(
	# 		pretrained=False,
	# 		num_classes=args.embedding_size,
	# 	)
	# elif args.arch == 'iresnet50':
	# 	features = iresnet.iresnet50(
	# 		pretrained=False,
	# 		num_classes=args.embedding_size,
	# 	)
	# elif args.arch == 'iresnet100':
	# 	features = iresnet.iresnet100(
	# 		pretrained=False,
	# 		num_classes=args.embedding_size,
	# 	)
	else:
		raise ValueError()
	return features


class NetworkBuilder_inf(nn.Module):
	def __init__(self, args):
		super(NetworkBuilder_inf, self).__init__()
		self.features = load_features(args)
	
	def forward(self, input):
		# add Fp, a pose feature
		x = self.features(input)
		return x


def load_dict_inf(args, model):
	if os.path.isfile(args['resume']):
		cprint('=> loading pth from {} ...'.format(args['resume']))
		# if args.cpu_mode:
		# 	checkpoint = torch.load(args['resume'], map_location=torch.device("cpu"))
		# else:
		checkpoint = torch.load(args['resume'], weights_only=True)
		_state_dict = clean_dict_inf(model, checkpoint['state_dict'])
		model_dict = model.state_dict()
		model_dict.update(_state_dict)
		model.load_state_dict(model_dict)
		# delete to release more space
		del checkpoint
		del _state_dict
	else:
		sys.exit("=> No checkpoint found at '{}'".format(args.resume))
	return model


def clean_dict_inf(model, state_dict):
	_state_dict = OrderedDict()
	for k, v in state_dict.items():
		# # assert k[0:1] == 'features.module.'
		new_k = 'features.'+'.'.join(k.split('.')[2:])
		if new_k in model.state_dict().keys() and \
				v.size() == model.state_dict()[new_k].size():
			_state_dict[new_k] = v
		# assert k[0:1] == 'module.features.'
		new_kk = '.'.join(k.split('.')[1:])
		if new_kk in model.state_dict().keys() and \
				v.size() == model.state_dict()[new_kk].size():
			_state_dict[new_kk] = v
	num_model = len(model.state_dict().keys())
	num_ckpt = len(_state_dict.keys())
	if num_model != num_ckpt:
		sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
			num_model, num_ckpt))
	return _state_dict


def builder_inf(args):
	model = NetworkBuilder_inf(args)
	# Used to run inference
	model = load_dict_inf(args, model)
	return model





class MagFaceLoss():
	def __init__(self):
		self.args = {}
		self.args['arch'] = 'iresnet100'
		self.args['embedding_size'] = 512
		self.args['resume'] = '/CT/VORF_GAN/static00/pretrained/magface/magface_epoch_00025.pth'
	
	def build_model(self):
		model = builder_inf(self.args)
		model = model.cuda()
		model.eval()
		return model

def img2magfaceInput(cv2_img, device='cuda'):
	"""
	Convert a cv2 image (numpy array in H x W x C) to a tensor of shape (H1 x W1 x C).
	Args:
		cv2_img (np.array): Input image with shape (H, W, C) and pixel values in 0–255.
		device (str): The device to move the tensor to ('cuda' or 'cpu').

	Returns:
		torch.Tensor: Processed image tensor with shape (1, C, H, W) and values in [-1,1].
	"""
	# Resize the image
	
	cv2_img = cv2.resize(cv2_img, (112, 112)).astype(np.float32)
	
	# Define a transform that converts to tensor and normalizes
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0., 0. ,0.],
			std=[1., 1. ,1.])
	])
	
	# Apply transform and add a batch dimension
	tensor_img = transform(cv2_img).unsqueeze(0).to(device)
	return tensor_img



	
	
	

def compute_id_loss(pred_img, gt_img):
	# Using opencv resize the image to 112x112
	
	# Input
	pred_img = process_cv2_image(pred_img, device='cuda')
	gt_img = process_cv2_image(gt_img, device='cuda')
	# gt_img = torch.Tensor((gt_img / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
	# breakpoint()
	# pred_img = torch.Tensor((pred_img / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))).cuda()
	
	print(pred_img.shape, gt_img.shape)


	embedding_feat = model(input[0])
	print(embedding_feat.shape)

def test():
	model = builder_inf(args)
	model = torch.nn.DataParallel(model)
	if not args.cpu_mode:
		model = model.cuda()
	model.eval()
