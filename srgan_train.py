# Libraries
import os.path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import LoadTrainDatasetFromFolder, LoadValidDatasetFromFolder
from tqdm import tqdm

from models import Generator, Discriminator
from srgan_loss import GeneratorLoss

from math import log10, sqrt
from pytorch_msssim import ssim

# Data Locations
train_data_path = "./train_data/VOC2012_train_HR"
valid_data_path = "./train_data/VOC2012_valid_HR"

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
batch_size = 16
crop_size = 96
num_epochs = 5
upscale_factor = 4
learning_rate = 0.0001

# Initialize Networks & Losses
netG = Generator().to(device)
netD = Discriminator().to(device)
lossG = GeneratorLoss().to(device)
bce_loss = nn.BCELoss()

print("Number of generator parameters: ", sum(param.numel() for param in netG.parameters()))
print("Number of discriminator parameters: ", sum(param.numel() for param in netD.parameters()))

# Load Data
train_set = LoadTrainDatasetFromFolder(dataset_dir=train_data_path, crop_size=crop_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)  # If on Kaggle, set num_workers=2

valid_set = LoadValidDatasetFromFolder(dataset_dir=valid_data_path, upscale_factor=4)
valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=True)  # If on Kaggle, set num_workers=2

# Initialize Optimizer
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.9, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Create where results will be saved
results = {"d_loss": [], "g_loss": [], "d_score": [], "g_score": [], "mse": [], "psnr": [], "ssim": []}
if not os.path.exists("./train_results/"):
    os.makedirs("./train_results/")

# Train Network
for epoch in range(1, num_epochs + 1):
    train_bar = tqdm(train_loader)

    netG.train()
    netD.train()
    train_set_size = 0
    train_results = {"d_loss": 0, "g_loss": 0, "d_score": 0, "g_score": 0}
    for data, target in train_bar:
        batch_size = data.size(0)
        train_set_size += data.size(0)
        # Get data to cuda() and enable gradient
        data.requires_grad_(True)
        target.requires_grad_(True)
        data = data.to(device=device)
        target = target.to(device=device)

        ##############################
        # TRAINING THE DISCRIMINATOR #
        ##############################
        # Forward Pass
        netD.zero_grad()
        real_images = target
        fake_images = netG(data)
        real_output = netD(real_images)
        fake_output = netD(fake_images)

        # Backward Pass
        discriminator_real_loss = bce_loss(real_output, torch.ones_like(real_output))
        discriminator_fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_loss.backward(retain_graph=True)

        # Weight Update (Gradient step)
        optimizerD.step()

        ##########################
        # TRAINING THE GENERATOR #
        ##########################
        # Forward Pass
        netG.zero_grad()
        real_images = target
        fake_images = netG(data)
        fake_output = netD(fake_images)

        # Backward Pass
        generator_loss = lossG(fake_output, fake_images, real_images)
        generator_loss.backward()

        # Weight Update (Gradient step)
        optimizerG.step()

        # Training Results
        train_results["d_loss"] += discriminator_loss.item()
        train_results["g_loss"] += generator_loss.item()
        train_results["d_score"] += real_output.mean().item()
        train_results["g_score"] += fake_output.mean().item()

        train_bar.set_description(desc="[%d/%d]" % (epoch, num_epochs))

    # Saving training results
    results["d_loss"].append(train_results["d_loss"] / train_set_size)
    results["g_loss"].append(train_results["g_loss"] / train_set_size)
    results["d_score"].append(train_results["d_score"] / train_set_size)
    results["g_score"].append(train_results["g_score"] / train_set_size)

    netG.eval()
    with torch.no_grad():
        valid_bar = tqdm(valid_loader)
        valid_set_size = 0
        valid_results = {"mse": 0, "ssim": 0, "psnr": 0}
        for lr_image, hr_image in valid_bar:
            valid_set_size += lr_image.size(0)
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            sr_image = netG(lr_image).to(device)

            # Image Similarity Metrics
            mse_val = (((sr_image - hr_image) ** 2).data.mean()).item()
            ssim_val = ssim(sr_image, hr_image).item()
            psnr_val = 100 if mse_val == 0 else (20 * log10(255.0 / sqrt(mse_val)))

            # Validation Results
            valid_results["mse"] += mse_val
            valid_results["ssim"] += ssim_val
            valid_results["psnr"] += psnr_val

            valid_bar.set_description(desc="[Validation] PSNR: %.5f SSIM: %.5f" % (valid_results["psnr"] / valid_set_size, valid_results["ssim"] / valid_set_size))

        # Saving validation results
        results["mse"].append(valid_results["mse"] / valid_set_size)
        results["ssim"].append(valid_results["ssim"] / valid_set_size)
        results["psnr"].append(valid_results["psnr"] / valid_set_size)

    # Save every fifth model
    if epoch % 5 == 0:
        models_folder_path = "./train_results/models/"
        if not os.path.exists(models_folder_path):
            os.makedirs(models_folder_path)
        torch.save(netG.state_dict(), (models_folder_path + "SRGAN_netG_epoch_%d.pth") % epoch)
        torch.save(netD.state_dict(), (models_folder_path + "SRGAN_netD_epoch_%d.pth") % epoch)

# Save statistical results from during training
statistics_folder_path = "./train_results/statistics/"
if not os.path.exists(statistics_folder_path):
    os.makedirs(statistics_folder_path)
model_statistics = pd.DataFrame(
    data={"Discriminator Loss": results["d_loss"], "Generator Loss": results["g_loss"], "Discriminator Score": results["d_score"],
          "Generator Score": results["g_score"], "MSE": results["mse"], "PSNR": results["psnr"], "SSIM": results["ssim"]},
    index=range(1, num_epochs + 1)
)
model_statistics.to_csv(statistics_folder_path + 'SRGAN_train_results.csv', index_label='Epoch')
