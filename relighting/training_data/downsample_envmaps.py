import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import cv2
import glob
from natsort import natsorted

if __name__ == '__main__':
	emap_path = '/CT/VORF_GAN4/static00/datasets/env_maps/indoor/'
	output_emap_path = '/CT/VORF_GAN4/static00/datasets/env_maps/indoor_ds/'
	
	emap_fnames_list = natsorted(glob.glob(emap_path + '*.exr'))
	
	for fname in emap_fnames_list:
		emap = cv2.imread(fname, -1)
		emap = cv2.resize(emap, (20, 10))
		cv2.imwrite(os.path.join(output_emap_path, os.path.basename(fname)), emap)

