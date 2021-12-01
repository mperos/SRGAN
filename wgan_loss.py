import torch
from torch import nn
from torchvision.models import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg_network = vgg19(pretrained=True).features[:36].eval().to(device)
        for param in vgg_network.parameters():
            param.requires_grad = False
        self.vgg_network = vgg_network
        self.mse_loss = nn.MSELoss()

    def forward(self, fake_output, fake_images, real_images):
        adversarial_loss = -torch.mean(fake_output)
        vgg_loss = self.mse_loss(self.vgg_network(fake_images), self.vgg_network(real_images))
        l2_loss = self.mse_loss(fake_images, real_images)
        return l2_loss + 0.001 * adversarial_loss + 0.006 * vgg_loss


def gradient_penalty(critic, real_images, fake_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, c, h, w = real_images.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real_images * epsilon + fake_images * (1 - epsilon)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty_value = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty_value
