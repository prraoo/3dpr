import os
import sys
import glob
from natsort import natsorted
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

base_path = sys.argv[1]
subject_id = sys.argv[2]
input_path = os.path.join(base_path, subject_id, 'crop')
mask_path = os.path.join(base_path, subject_id, 'mask_rmbg2')

images_path = natsorted(glob.glob1(input_path, f'*'))
os.makedirs(mask_path, exist_ok=True)

for fname in images_path:
    image = Image.open(os.path.join(input_path, fname))
    
    input_images = transform_image(image).unsqueeze(0).to('cuda')
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask.save(os.path.join(mask_path, fname))