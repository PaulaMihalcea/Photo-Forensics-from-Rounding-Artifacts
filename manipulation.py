import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import get_image_size, load_image


# Define random manipulation region
def manipulate_image(img_path, manipulation):
    # Load image
    img = load_image(img_path)
    img_height, img_width = get_image_size(img)

    # Randomly choose manipulation region
    region_size = random.choice([512, 256, 128, 64])
    x = random.randint(0, img_width - region_size)
    y = random.randint(0, img_height - region_size)

    # Select region to manipulate
    manip_region = img[y:y+region_size, x:x+region_size]

    # Apply manipulation
    if manipulation == 'copy_move':
        copy_move(manip_region)
    elif manipulation == 'median_filter':
        manip_region = median_filter(manip_region, 3)
    elif manipulation == 'rotation':
        rotate(manip_region, img) # TODO
    elif manipulation == 'content_aware_fill':
        content_aware_fill(manip_region, img)  # TODO

    # Generate manipulated image
    manip_img = 0  # TODO

    # Generate ground truth image
    ground_truth = 0  # TODO

    return manip_img, ground_truth


# Copy-move
def copy_move(img):
    return


# Median filter
def median_filter(img, size):
    return cv2.medianBlur(img, size)


# Copy-move
def rotate(img):
    return


# Copy-move
def content_aware_fill(img):
    return
