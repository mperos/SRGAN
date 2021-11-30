# Libraries
import torch
import torch.nn as nn


# Model Classes
class ResidualBlock(nn.Module):
    """
    SRGAN Paper - Residual Block:
    Conv2d(n64, k3, s1) -> BatchNorm() -> PReLu() -> Conv2d(n64, k3, s1) -> BatchNorm()
    """
    def __init__(self, channels):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(num_features=channels)
        self.PReLU = nn.PReLU()
        self.Conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        output = self.Conv1(x)
        output = self.BatchNorm1(output)
        output = self.PReLU(output)
        output = self.Conv2(output)
        output = self.BatchNorm2(output)
        return output + x


class UpsampleBlock(nn.Module):
    """
    SRGAN Paper - Upsample Block:
    Conv2d(n256, k3, s1) -> Pixel Shuffle x2 -> PReLu()
    """
    def __init__(self, channels, upscale_factor):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels=channels, out_channels=channels * (upscale_factor ** 2), kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        output = self.Conv(x)
        output = self.PixelShuffle(output)
        output = self.PReLU(output)
        return output


class ConvolutionBlock(nn.Module):
    """
    SRGAN Paper - Convolution Block:
    Conv2d() -> BatchNorm() -> LeakyReLu()
    """
    def __init__(self, in_channels, out_channels, stride_size):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride_size, padding=1)
        self.BatchNorm = nn.BatchNorm2d(num_features=out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.Conv(x)
        output = self.BatchNorm(output)
        output = self.LeakyReLU(output)
        return output


class Generator(nn.Module):
    """
    SRGAN Paper - Generator Network:
    Input -> Conv2d(n64, k9, s1) -> PRelu() -> 16x ResidualBlock ->Conv2d(n64, k3, s1) ->
    BatchNorm() -> SkipConn -> 2x UpsampleBlock -> Conv2d(n3, k9, s1) -> Output
    """
    def __init__(self):
        super().__init__()
        self.InputBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 9), stride=(1, 1), padding=4),
            nn.PReLU()
        )

        # Book "Hands-On Generative Adversarial Networks with PyTorch 1.x" uses only
        # 5 Residual Blocks, while original SRGAN paper uses 16 Residual Blocks.
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])

        self.ConvBatchNormBlock = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64)
        )

        self.UpsampleBlocks = nn.Sequential(
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(9, 9), stride=(1, 1), padding=4)
        )

    def forward(self, x):
        input_block = self.InputBlock(x)
        residual = self.ResidualBlocks(input_block)
        residual = self.ConvBatchNormBlock(residual)
        output = self.UpsampleBlocks(input_block + residual)

        # When using torch.tanh(output) we get weird artifacts when changing ToPILImage() because
        # ToPILImage() doesn't clip negative values (which do rarely occur), while plt clips those values.
        return (torch.tanh(output) + 1) / 2


class Discriminator(nn.Module):
    """
    SRGAN Paper - Discriminator Network:
    Input -> Conv2d(n64, k3, s1) -> LeakyReLu() -> Conv2d(n64, k3, s2) -> Conv2d(n128, k3, s1) ->
    Conv2d(n128, k3, s2) -> Conv2d(n256, k3, s1) -> Conv2d(n256, k3, s2) -> Conv2d(n512, k3, s1) ->
    Conv2d(n512, k3, s2) -> DenseLayer(1024) ->LeakyReLu() -> DenseLayer(1) -> Sigmoid() -> Output
    """

    def __init__(self):
        super().__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2)
        )
        self.Block2 = ConvolutionBlock(64, 64, 2)
        self.Block3 = ConvolutionBlock(64, 128, 1)
        self.Block4 = ConvolutionBlock(128, 128, 2)
        self.Block5 = ConvolutionBlock(128, 256, 1)
        self.Block6 = ConvolutionBlock(256, 256, 2)
        self.Block7 = ConvolutionBlock(256, 512, 1)
        self.Block8 = ConvolutionBlock(512, 512, 2)  # 512 x 64 x 64 (512 channels, 64x64 matrix)
        self.Block9 = nn.Sequential(
            # torchvision models use adaptive pooling layers after the feature extractor and before feeding
            # the activation to the first linear layer to allow different input shapes.
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=512 * 2 * 2, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        output = self.Block1(x)
        output = self.Block2(output)
        output = self.Block3(output)
        output = self.Block4(output)
        output = self.Block5(output)
        output = self.Block6(output)
        output = self.Block7(output)
        output = self.Block8(output)
        output = self.Block9(output)

        return output


class Critic(nn.Module):
    """
    Critic Network: (Except for the output, same as discriminator in SRGAN Paper)
    Input -> Conv2d(n64, k3, s1) -> LeakyReLu() -> Conv2d(n64, k3, s2) -> Conv2d(n128, k3, s1) ->
    Conv2d(n128, k3, s2) -> Conv2d(n256, k3, s1) -> Conv2d(n256, k3, s2) -> Conv2d(n512, k3, s1) ->
    Conv2d(n512, k3, s2) -> DenseLayer(1024) ->LeakyReLu() -> DenseLayer(1) -> Sigmoid() -> Output
    """

    def __init__(self):
        super().__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2)
        )
        self.Block2 = ConvolutionBlock(64, 64, 2)
        self.Block3 = ConvolutionBlock(64, 128, 1)
        self.Block4 = ConvolutionBlock(128, 128, 2)
        self.Block5 = ConvolutionBlock(128, 256, 1)
        self.Block6 = ConvolutionBlock(256, 256, 2)
        self.Block7 = ConvolutionBlock(256, 512, 1)
        self.Block8 = ConvolutionBlock(512, 512, 2)
        self.Block9 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        output = self.Block1(x)
        output = self.Block2(output)
        output = self.Block3(output)
        output = self.Block4(output)
        output = self.Block5(output)
        output = self.Block6(output)
        output = self.Block7(output)
        output = self.Block8(output)
        output = self.Block9(output)

        return output
