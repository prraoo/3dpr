import sys
import os
# Add project root to Python path to allow imports from render_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import argparse
import cv2
import numpy as np
import shutil
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import pillow_avif
import torch
import json
from torchvision.transforms.functional import pil_to_tensor

# Enable openexr support
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from render_utils.image import (
    srgb2linear, linear2srgb
)
from render_utils import envmap
from torchvision.utils import save_image

def project_envmap_to_olat_solid_angle(
    index_map: np.ndarray,
    envmap: np.ndarray
) -> np.ndarray:
    """
    Projects a latlong envmap onto an OLAT basis, weighting each pixel
    by its corresponding solid angle to correct for latlong projection distortion.

    Args:
        index_map (np.ndarray): A 2D integer array (H, W) where each pixel's value
                                is the index of the light/Voronoi cell it belongs to.
        envmap (np.ndarray): The environment map image. Can be grayscale (H, W) or
                             color (H, W, C). Expected to be in float format.

    Returns:
        np.ndarray: An array of projection coefficients of shape (num_lights, num_channels).
    """
    # Ensure the environment map is in a floating-point format
    if not np.issubdtype(envmap.dtype, np.floating):
        envmap = envmap.astype(np.float32)

    # Validate input shapes
    if index_map.shape != envmap.shape[:2]:
        raise ValueError("The height and width of the index_map and envmap must match.")

    height, width = envmap.shape[:2]

    # --- 1. Calculate solid angle weights for each pixel row ---
    # The solid angle is proportional to sin(theta), where theta is the polar angle.
    # We calculate theta for the center of each pixel row.
    # theta = 0 at the top (north pole), pi at the bottom (south pole).
    theta = (np.arange(height) + 0.5) * (np.pi / height)
    solid_angle_row_weights = np.sin(theta) * 2.0 * np.pi / width

    # --- 2. Apply weights to the environment map ---
    # We use NumPy broadcasting to multiply each row of the envmap by its weight.
    # The weights array is reshaped to (H, 1) or (H, 1, 1) to enable this.
    if envmap.ndim == 3:  # Color image (H, W, C)
        # Reshape weights to (H, 1, 1) to broadcast across width and channels
        weighted_envmap = envmap * solid_angle_row_weights[:, np.newaxis, np.newaxis]
    else:  # Grayscale image (H, W)
        # Reshape weights to (H, 1) to broadcast across width
        weighted_envmap = envmap * solid_angle_row_weights[:, np.newaxis]

    # --- 3. Sum the weighted pixel values for each Voronoi cell ---
    num_lights = int(np.max(index_map)) + 1
    flat_indices = index_map.flatten()

    # Handle both grayscale and color weighted images
    if envmap.ndim == 2:
        flat_weighted_envmap = weighted_envmap.flatten()
        coeffs = np.bincount(flat_indices, weights=flat_weighted_envmap, minlength=num_lights)
        return coeffs[:, np.newaxis]  # Return as (num_lights, 1)
    
    elif envmap.ndim == 3:
        num_channels = envmap.shape[2]
        coeffs = np.zeros((num_lights, num_channels), dtype=np.float32)
        flat_weighted_envmap = weighted_envmap.reshape(-1, num_channels)
        
        # Apply bincount for each channel separately
        for c in range(num_channels):
            coeffs[:, c] = np.bincount(
                flat_indices,
                weights=flat_weighted_envmap[:, c],
                minlength=num_lights
            )
        return coeffs
    else:
        raise ValueError("envmap must be a 2D (grayscale) or 3D (color) array.")

def load_light_pattern(sub: str) -> List[Tuple[int]]:
    # \CT\RelightAvatar2\nobackup\tmp\datasets\goliath
    light_pattern_path = Path("/CT/RelightAvatar2/nobackup/tmp/datasets/goliath") / sub / "StaticHead" / "lights" / "light_pattern_per_frame.json"
    with open(light_pattern_path, "r") as f:
        return json.load(f)

def load_light_pattern_meta(sub: str) -> Dict[str, Any]:
    light_pattern_path = Path("/CT/RelightAvatar2/nobackup/tmp/datasets/goliath") / sub / "StaticHead" / "lights" / "light_pattern_metadata.json"
    with open(light_pattern_path, "r") as f:
        return json.load(f)

def load_color_calib():
    mat = np.loadtxt(Path("/CT/RelightAvatar2/nobackup/tmp/datasets/goliath") / "shared" / "mpi" / "colorcalib.mat", dtype=np.float32)
    return torch.from_numpy(mat)

def color_calib(image):
    mat = load_color_calib()

    # black level
    image -= mat[:3, 3][..., None, None]

    # color calib
    image = torch.sum(mat[:3, :3][..., None, None] * image[None], dim=1)

    return image

def load_image(sub: str, frame: int, camera: str) -> Image:
    ext = "avif"
    reexpose = 2.0 if sub.startswith("ID2") and int(sub[3:]) >= 88 and int(sub[3:]) <= 116 else 1.0
    img_load_func = lambda x: color_calib(reexpose * srgb2linear(pil_to_tensor(Image.open(x)) / 255.0))

    return img_load_func(Path("/CT/RelightAvatar2/nobackup/tmp/datasets/goliath") / sub / "StaticHead" / "image" / f"{camera}/{frame:06d}.{ext}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envpath", type=str, default="/CT/RelightAvatar/nobackup/tmp/latlongs")
    parser.add_argument("--indexmap", type=str, default="relighting/evaluation_data/voronoi_indices.exr")
    parser.add_argument("--envname", type=str, default="grace_cathedral")
    parser.add_argument("--envscale", type=float, default=0.01)
    parser.add_argument("--sub", type=str, default="ID20010")
    parser.add_argument("--outdir", type=str, default="/CT/VORF_GAN5/nobackup/datasets/evaluation/CVPR26-debug_00/")
    parser.add_argument("--cycle", type=int, default=256)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()
    return args

def main(args):
    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    indexmap = cv2.imread(args.indexmap, cv2.IMREAD_UNCHANGED).astype(np.int32)
    env_image = cv2.imread(os.path.join(args.envpath, f"{args.envname}.exr"), cv2.IMREAD_UNCHANGED).clip(min=0.0)[..., :3][..., ::-1]
    assert indexmap.shape[:2] == env_image.shape[:2]
    env_image = torch.from_numpy(env_image.copy()).float().permute((2, 0, 1)).contiguous()

    images = torch.zeros(331, 3, 1024, 1024)
    light_pattern = load_light_pattern(args.sub)
    for frame in range(348):
        light_index = light_pattern[frame][1] - 1
        if light_index < 0:
            continue
        images[light_index] = load_image(args.sub, frame, "Cam06").clamp(min=0.0)
    
    for i in range(args.cycle):
        index = (i + args.start_index) % args.cycle
        rot_y = 2.0 * np.pi * index / args.cycle
        axis = torch.Tensor([0.0, 1.0, 0.0])
        quat = torch.Tensor(axis.tolist() + [rot_y]).float()
        quat = torch.nn.functional.normalize(quat[0:3], dim=0) * quat[3]
        rot_mat = envmap.rvec_to_R(quat)
        new_env = envmap.rotate_envmap_mat(env_image, rot_mat)
        light_intensities = project_envmap_to_olat_solid_angle(indexmap, new_env.permute(1, 2, 0).numpy()) * args.envscale
        assert (light_intensities[:, :3] >= 0).all()
        ref = (images * torch.from_numpy(light_intensities[..., None, None])).sum(0)

        # cv2.imwrite(os.path.join(args.outdir, "final_image.exr"),final_image.permute(1, 2, 0).numpy().astype(np.float32)[..., ::-1])
        save_image(linear2srgb(ref), os.path.join(args.outdir, f"{i}.png"))



if __name__ == "__main__":
    args = parse_args()
    main(args)