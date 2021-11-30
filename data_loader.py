# Libraries
from os import listdir
from os.path import join

from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize, ToTensor, ToPILImage

from PIL import Image

# Helper Functions
def is_image(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def get_valid_crop_size(crop_size, upscale_factor):
    """ If we upscale by upscale_factor, then hr_image needs to be
    dividable by upscale_factor to have a valid lr_image. """
    return crop_size - (crop_size % upscale_factor)


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def get_hr_train_image(crop_size):
    return Compose([
        RandomCrop(crop_size, pad_if_needed=True),
        ToTensor()
    ])


def get_lr_train_image(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def get_hr_valid_image(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])


def get_lr_valid_image(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


# Classes
class LoadTrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super().__init__()
        self.image_filenames = [join(dataset_dir, filename) for filename in listdir(dataset_dir) if is_image(filename)]
        crop_size = get_valid_crop_size(crop_size, upscale_factor)
        self.get_hr_train_image = get_hr_train_image(crop_size)
        self.get_lr_train_image = get_lr_train_image(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.get_hr_train_image(Image.open(self.image_filenames[index]))
        lr_image = self.get_lr_train_image(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class LoadValidDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super().__init__()
        self.image_filenames = [join(dataset_dir, filename) for filename in listdir(dataset_dir) if is_image(filename)]
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        w, h = image.size
        crop_size = get_valid_crop_size(min(w, h), self.upscale_factor)
        get_hr_image = get_hr_valid_image(crop_size)
        get_lr_image = get_lr_valid_image(crop_size, self.upscale_factor)
        hr_image = get_hr_image(image)
        lr_image = get_lr_image(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class LoadTestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super().__init__()
        lr_image_dataset = dataset_dir + "lr_images/"
        hr_image_dataset = dataset_dir + "hr_images/"
        self.upscale_factor = upscale_factor
        self.lr_image_filenames = [join(lr_image_dataset, filename) for filename in listdir(lr_image_dataset) if is_image(filename)]
        self.hr_image_filenames = [join(hr_image_dataset, filename) for filename in listdir(hr_image_dataset) if is_image(filename)]

    def __getitem__(self, index):
        image_name = self.hr_image_filenames[index].split('/')[-1]
        lr_image = ToTensor()(Image.open(self.lr_image_filenames[index]))
        hr_image = ToTensor()(Image.open(self.hr_image_filenames[index]))
        h, w = lr_image.shape[1], lr_image.shape[2]
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_rescaled_image = hr_scale(hr_image)  # Image with same dimensions as upscaled lr_image
        return image_name, lr_image, hr_rescaled_image, hr_image

    def __len__(self):
        return len(self.lr_image_filenames)
