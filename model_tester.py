import os
import numpy as np
import pandas as pd

import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Generator
from data_loader import LoadTestDatasetFromFolder, display_transform
from torchvision.transforms import ToTensor

from math import log10, sqrt
from pytorch_msssim import ssim

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

# Iterate over test image sets
image_sets = ["Set5", "Set14", "BSD100"]
for image_set in image_sets:
    test_data_path = "./test_data/" + image_set + "/"
    # Upscale factor set to 4 because of the generators design
    test_set = LoadTestDatasetFromFolder(test_data_path, upscale_factor=4)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc="[Getting test images]")

    test_images_folder_path = "./test_results/" + image_set + "/"
    if not os.path.exists(test_images_folder_path):
        os.makedirs(test_images_folder_path)

    num_of_images = 0
    results = {"mse": [], "psnr": [], "ssim": []}

    for image_name, lr_image, hr_rescaled_image, hr_image in test_bar:
        num_of_images += lr_image.size(0)
        image_name = image_name[0]

        lr_image = lr_image.to(device)
        hr_rescaled_image = hr_rescaled_image.to(device)
        hr_image = hr_image.to(device)
        sr_image = model(lr_image)

        mse_val = ((hr_rescaled_image - sr_image) ** 2).data.mean().item()
        psnr_val = 10 * log10(1 / mse_val)
        ssim_val = ssim(sr_image, hr_image, data_range=1).item()

        results["mse"].append(mse_val)
        results["psnr"].append(psnr_val)
        results["ssim"].append(ssim_val)

        test_images = torch.stack([
            display_transform()(lr_image.squeeze(0)),
            display_transform()(hr_image.squeeze(0)),
            display_transform()(sr_image.squeeze(0))
        ])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, test_images_folder_path + image_name, padding=5)

    statistics_folder_path = "./test_results/statistics/"
    if not os.path.exists(statistics_folder_path):
        os.makedirs(statistics_folder_path)
    model_statistics = pd.DataFrame(
        data={"MSE": results["mse"], "PSNR": results["psnr"], "SSIM": results["ssim"]},
        index=range(1, num_of_images + 1)
    )
    model_statistics.to_csv(statistics_folder_path + image_set + '_test_results.csv', index_label='Image')
