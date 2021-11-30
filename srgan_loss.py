import torch
from torch import nn
from torchvision.models import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg_network = vgg19(pretrained=True).features[:36].eval().to(device)  # :35?
        for param in vgg_network.parameters():
            param.requires_grad = False
        self.vgg_network = vgg_network
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_output, fake_images, real_images):
        adversarial_loss = self.bce_loss(fake_output, torch.ones_like(fake_output))
        vgg_loss = self.mse_loss(self.vgg_network(fake_images), self.vgg_network(real_images))
        l2_loss = self.mse_loss(fake_images, real_images)
        return 0.001 * adversarial_loss + 0.006 * vgg_loss + l2_loss
