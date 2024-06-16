from typing import Sequence
import random

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Pad, Resize


class FixedAspectResize(nn.Module):
    def __init__(self, target_size):
        super().__init__()

        self.target_size = target_size
        self.aspect_ratio = target_size[0] / target_size[1] if isinstance(target_size, Sequence) else 1

    def forward(self, image):
        """The image is expected to be a binarized tensor or PIL.Image."""
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        img_aspect_ratio = h / w
        if img_aspect_ratio > self.aspect_ratio:
            new_w = int(w * self.aspect_ratio)
            pad_size = new_w - w
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            pad_top, pad_bottom = 0, 0
        else:
            new_h = int(w / self.aspect_ratio)
            pad_size = new_h - h
            pad_top = pad_size // 2
            pad_bottom = pad_size - pad_top
            pad_left, pad_right = 0, 0

        padding = Pad(padding=[pad_left, pad_top, pad_right, pad_bottom])
        resizing = Resize(self.target_size)

        new_image = resizing(padding(image))

        return new_image


class RandomSpots(nn.Module):
    def __init__(self, spots_range, w_range, h_range):
        super().__init__()
        
        self.spots_range = spots_range
        self.w_range = w_range
        self.h_range = h_range

    def forward(self, image):
        """The image is expected to be a binarized np.ndarray or PIL.Image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape

        num_spots = random.randint(*self.spots_range)
        angle = random.randint(0, 360)
        for _ in range(num_spots):
            center_x = random.randint(0, w - 1)
            center_y = random.randint(0, h - 1)
            spot_w = random.randint(*self.w_range)
            spot_h = random.randint(*self.h_range)
            
            new_image = cv2.ellipse(
                image, 
                center=(center_x, center_y), 
                axes=(spot_w, spot_h), 
                angle=angle, 
                startAngle=0,
                endAngle=360,
                color=255, 
                thickness=cv2.FILLED
            )

        return new_image


class Binarization(nn.Module):
    def __init__(self, inv_color=False):
        super().__init__()

        self.inv_color = inv_color

    def forward(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.inv_color:
            gray_image = 255 - gray_image
        
        blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
        background = cv2.blur(blurred_image, (11, 11))
        foreground = cv2.subtract(blurred_image, background)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(12, 12))
        enhanced_foreground = clahe.apply(foreground)

        _, binary_image = cv2.threshold(
            src=enhanced_foreground, 
            thresh=0, maxval=255, 
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary_image
