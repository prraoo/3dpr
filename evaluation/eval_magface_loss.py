import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set(style="white")

def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
def show(idx_):
    imgname = imgnames[idx_]
    img = cv2.imread(imgname)
    imshow(img)
    print(img_2_mag[imgname], imgname)
    

def compute_features(img_list, feat_list):
    # with open('/CT/VORF_GAN3/work/code/MagFace/inference/toy_imgs/feats.list', 'r') as f:
    with open(feat_list, 'r') as f:
        lines = f.readlines()
    
    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        feat = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feat)
        img_2_feats[imgname] = feat/mag
        img_2_mag[imgname] = mag
    
    imgnames = list(img_2_mag.keys())
    mags = [img_2_mag[imgname] for imgname in imgnames]
    feats = [img_2_feats[imgname] for imgname in imgnames]
    
    return imgnames, mags, feats
    # sort_idx = np.argsort(mags)
    

def plot_images(imgnames, mags, sort_idx, save_path):
    H, W = 112, 112
    if len(imgnames) < 7:
        NH, NW = 1, len(imgnames)
    else:
        NH, NW = 3, len(imgnames) //3
    canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

    for i, ele in enumerate(sort_idx):
        imgname = os.path.join('/CT/VORF_GAN3/work/code/MagFace/inference/', imgnames[ele])
        img = cv2.imread(imgname)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (112, 112))
        canvas[int(i / NW) * H: (int(i / NW) + 1) * H, (i % NW) * W: ((i % NW) + 1) * W, :] = img
      
    plt.figure(figsize=(20, 20))
    print([float('{0:.2f}'.format(mags[idx_])) for idx_ in sort_idx])
    plt.imsave(f"{save_path}.jpg", canvas)

def calculate_projection(feats, sort_idx):
    similarity_loss = []
    for i, ele in enumerate(sort_idx):
        similarity = np.dot(feats[0], feats[ele].T)
        similarity_loss.append(similarity)
    
    print(np.array(similarity_loss))
    print(np.mean(np.array(similarity_loss[1:])))

        # print(imgnames[ele], feats[ele])
    
#
# feats = np.array([img_2_feats[imgnames[ele]] for ele in sort_idx])
# sim_mat = np.dot(feats, feats.T)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax = sns.heatmap(sim_mat, cmap="PuRd", annot=True)

if __name__ == '__main__':
    imgs_list_path = sys.argv[1]
    feats_list_path = sys.argv[2]
    assert os.path.isfile(imgs_list_path), f"{imgs_list_path} does not exist!"
    assert os.path.isfile(feats_list_path), f"{feats_list_path} does not exist!"
    base_dir = sys.argv[3]
    fname = os.path.basename(imgs_list_path)
    
    image_names, magnitudes, features = compute_features(imgs_list_path, feats_list_path)
    # sorted_indices = np.argsort(magnitudes)
    sorted_indices = [idx for idx in range(len(magnitudes))]
    plot_images(image_names, magnitudes, sorted_indices, save_path=os.path.join(base_dir, fname))
    calculate_projection(features, sorted_indices)
    
    
    