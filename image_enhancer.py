import os

import torch
import torchvision.utils as utils

from models import Generator
from torchvision.transforms import ToTensor

from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create test results folder
if not os.path.exists("./test_results/"):
    os.makedirs("./test_results/")

# Choose model (input model name)
model_folder_path = "./trained_models/"
model_name = "WSRGAN_netG_epoch_100.pth"

model = Generator().eval()
model.load_state_dict(torch.load(model_folder_path + model_name, map_location=device))

# Choose image (set path to image)
image_path = "./test_data/SunHays80/lr_images/128_abbey_sun_asztnlqhlrvirneh.png"
image_name = "sr_" + image_path.split("/")[-1]

lr_image = ToTensor()(Image.open(image_path)).to(device)
lr_image = torch.unsqueeze(lr_image, 0)
sr_image = model(lr_image)

# Save enhanced image in enhanced_images folder
sr_image_folder_path = "./test_results/enhanced_images/"
if not os.path.exists(sr_image_folder_path):
    os.makedirs(sr_image_folder_path)
utils.save_image(sr_image, sr_image_folder_path + image_name, padding=5)
