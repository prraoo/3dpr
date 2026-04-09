# Pretrained Models

Place release checkpoints in this folder for inference.

Download placeholder:

- Google Drive: `TODO_ADD_GOOGLE_DRIVE_LINK`

Expected files:

- `ffhqrebalanced512-128.pkl`: pretrained 3D generator.
- `encoder_FFHQ.pt`: inversion encoder checkpoint.
- `afa_FFHQ.pt`: AFA checkpoint.

Reflectance checkpoints are typically produced by training and can stay outside this folder. The release inference wrapper [`scripts/infer_goae.sh`](/CT/VORF_GAN4/work/code/3dpr/scripts/infer_goae.sh) expects the reflectance checkpoint path as an explicit argument.
