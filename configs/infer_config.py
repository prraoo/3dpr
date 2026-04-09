import os
from distutils.util import strtobool
import argparse

from numpy.distutils.fcompiler import str2bool


def get_parser():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='configs/swinv2.yaml')
    parser.add_argument("--data", type=str, help='path to data directory', default='../example/real_person')
    parser.add_argument("--G_ckpt", type=str, help='path to generator model', default='../pretrained/ffhqrebalanced512-128.pkl')
    parser.add_argument("--E_ckpt", type=str, help='path to GOAE encoder checkpoint', default='../pretrained/encoder_FFHQ.pt')
    parser.add_argument("--R_ckpt", type=str, help='path to GOAE encoder checkpoint', default=' ')
    parser.add_argument("--AFA_ckpt", type=str, help='path to AFA model checkpoint', default='../pretrained/afa_FFHQ.pt')
    parser.add_argument("--outdir", type=str, help='path to output directory', default='../output/')
    parser.add_argument("--envmap_zspiral_path", type=str, help='path to output directory', default='/CT/VORF_GAN4/static00/datasets/OLAT_c2-Multiple-IDs/vorf_gan_config/envmap_zspiral_mpi')
    parser.add_argument("--envmap_dir", type=str, help='directory containing .exr environment maps', default=None)
    parser.add_argument("--light_dirs_path", type=str, help='path to MPI light direction text file or Weyrich light direction numpy file', default=None)
    
    # model
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--o_dim", type=int, default=12)
    parser.add_argument("--use_g_feat", type=bool, default=False)
    parser.add_argument("--use_viewdirs", type=strtobool)
    parser.add_argument("--arch", type=str, default='resnet')
    parser.add_argument("--act_fn", type=str, default='sigmoid')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", type=str, help="specify used cuda idx ", default='0')
    parser.add_argument("--use_sr_encoder", type=bool)

    # other
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--w_frames', type=int, default=240)
    parser.add_argument("--lightstage_res", type=str, default='half')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--emap_sample_fn", type=str, default='max')

    
    # render type
    parser.add_argument("--input_view", type=strtobool, default=False)
    parser.add_argument("--multi_view", type=strtobool, default=False)
    parser.add_argument("--video", type=str2bool, default=False)
    parser.add_argument("--shape", type=str2bool, default=False)
    parser.add_argument("--edit", type=str2bool, default=False)
    parser.add_argument("--add_envmap", type=strtobool, default=False)
    parser.add_argument('--render_mode', type=int, default=1)
    parser.add_argument('--olat_idx', type=int, default=None)

    # edit
    parser.add_argument("--edit_attr", type=str, help="editing attribute direction", default="glass")
    parser.add_argument("--alpha", type=float, help="editing alpha", default=1.0)

    # evaluation
    parser.add_argument("--eval_mode", type=str2bool, default=False)
    parser.add_argument('--input_cam', type=str, default='Cam07')
    parser.add_argument('--scan_name', type=str, default='ID00307')
    parser.add_argument('--num_emaps', type=int, default=3)
    parser.add_argument("--save_gt", action="store_true", default=False)
    parser.add_argument('--mode', type=str, default='quantitative')
    parser.add_argument('--save_nv', type=str2bool, default=False)
    
    # masking
    parser.add_argument("--use_mask",  type=str2bool, default=False)
    parser.add_argument("--gen_mask_mode", type=int, default=1, choices=[0, 1, 2],
                        help="Which method to generate mask: 0:No Mask, 1 SAM, 2 RMBG-v2")


    return parser
