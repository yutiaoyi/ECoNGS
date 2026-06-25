import torch
import torchvision
import torch.nn as nn
import numpy as np
from PIL import Image
import os

    
    

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def downsample_image(original_image, scale=1.0):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        return nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)

def resize_image(original_image, scale=1.0):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        _, image_height, image_width = original_image.shape
        resized = nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
        return nn.functional.interpolate(resized.unsqueeze(0), size=(image_height, image_width), mode='bilinear', align_corners=False).squeeze(0)

def blur_image(scale, transform):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return torch.nn.Identity()
    else:
        kernel_radius=int(transform.split("_")[1])
        cur_radius = round(kernel_radius*(1-scale))
        kernel_size=2*cur_radius+1
        sigma=float(transform.split("_")[2])
        if sigma == 0:
            cur_sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        else:
            cur_sigma = sigma*(1-scale)
        transform = torchvision.transforms.GaussianBlur((kernel_size,kernel_size), sigma=cur_sigma).cuda()
        return transform
    
def save_numpy_image(array: np.ndarray, save_path: str):
    """
    Save a numpy image array to a file.

    Args:
        array (np.ndarray): Image array of shape (H, W), (H, W, C), or (C, H, W)
        save_path (str): Output file path (e.g., "00001.png")
    """

    if array.dtype != np.uint8:
        array = np.round(array * 255).astype(np.uint8)

    img = Image.fromarray(array)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)