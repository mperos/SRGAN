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

from models import Generator, Critic
from wgan_loss import GeneratorLoss, gradient_penalty

from math import log10, sqrt
from pytorch_msssim import ssim

# Data Locations
train_data_path = "./train_data/DIV2K_train_HR"
valid_data_path = "./train_data/DIV2K_valid_HR"

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
batch_size = 32
crop_size = 96
num_epochs = 5
upscale_factor = 4
learning_rate = 0.0001
gp_lambda = 10

# Initialize Networks & Losses
netC = Critic().to(device)
netG = Generator().to(device)
lossG = GeneratorLoss().to(device)

print("Number of generator parameters: ", sum(param.numel() for param in netG.parameters()))
print("Number of discriminator parameters: ", sum(param.numel() for param in netC.parameters()))

# Load Data
train_set = LoadTrainDatasetFromFolder(dataset_dir=train_data_path, crop_size=crop_size, upscale_factor=4)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)  # If on Kaggle, set num_workers=2

valid_set = LoadValidDatasetFromFolder(dataset_dir=valid_data_path, upscale_factor=4)
valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=True)  # If on Kaggle, set num_workers=2

# Initialize optimizer as in WGAN-GP paper
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.0, 0.9))
optimizerC = optim.Adam(netC.parameters(), lr=learning_rate, betas=(0.0, 0.9))

# Create where results will be saved
results = {"c_loss": [], "g_loss": [], "c_score": [], "g_score": [], "mse": [], "psnr": [], "ssim": []}
if not os.path.exists("./train_results/"):
    os.makedirs("./train_results/")

# Train Network
for epoch in range(1, num_epochs + 1):
    train_bar = tqdm(train_loader)

    netC.train()
    netG.train()
    train_set_size = 0
    train_results = {"c_loss": 0, "g_loss": 0, "c_score": 0, "g_score": 0}
    for data, target in train_bar:
        batch_size = data.size(0)
        train_set_size += data.size(0)
        # Get data to cuda() and enable grad
        data.requires_grad_(True)
        target.requires_grad_(True)
        data = data.to(device=device)
        target = target.to(device=device)

        #############################
        #    TRAINING THE CRITIC    #
        #############################
        # Forward Pass
        netC.zero_grad()
        real_images = target
        fake_images = netG(data)
        real_output = netC(real_images)
        fake_output = netC(fake_images)

        # Backward Pass
        grad_penalty = gradient_penalty(netC, real_images, fake_images)
        # Torch optimizers minimize functions, thus minus of the equation
        critic_loss = (-(torch.mean(real_output) - torch.mean(fake_output)) + gp_lambda * grad_penalty)
        critic_loss.backward(retain_graph=True)

        # Weight Update
        optimizerC.step()

        ##########################
        # TRAINING THE GENERATOR #
        ##########################

        # Forward Pass
        netG.zero_grad()
        real_images = target
        fake_images = netG(data)
        fake_output = netC(fake_images)

        # Backward Pass
        generator_loss = lossG(fake_output, fake_images, real_images)
        generator_loss.backward()

        # Weight update
        optimizerG.step()

        # Training Results
        train_results["c_loss"] += critic_loss.item()
        train_results["g_loss"] += generator_loss.item()
        train_results["c_score"] += real_output.mean().item()
        train_results["g_score"] += fake_output.mean().item()

        train_bar.set_description(desc="[%d/%d]" % (epoch, num_epochs))

    # Saving training results
    results["c_loss"].append(train_results["c_loss"] / train_set_size)
    results["g_loss"].append(train_results["g_loss"] / train_set_size)
    results["c_score"].append(train_results["c_score"] / train_set_size)
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
        torch.save(netG.state_dict(), (models_folder_path + "WSRGAN_netG_epoch_%d.pth") % epoch)
        torch.save(netC.state_dict(), (models_folder_path + "WSRGAN_netC_epoch_%d.pth") % epoch)

# Save statistical results from during training
statistics_folder_path = "./train_results/statistics/"
if not os.path.exists(statistics_folder_path):
    os.makedirs(statistics_folder_path)
model_statistics = pd.DataFrame(
    data={"Critic Loss": results["c_loss"], "Generator Loss": results["g_loss"], "Critic Score": results["c_score"],
          "Generator Score": results["g_score"], "MSE": results["mse"], "PSNR": results["psnr"], "SSIM": results["ssim"]},
    index=range(1, num_epochs + 1)
)
model_statistics.to_csv(statistics_folder_path + 'WSRGAN_train_results.csv', index_label='Epoch')
