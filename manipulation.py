import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from utils import get_image_size, load_image


# Define random manipulation ROI
def manipulate_image(img_path, manipulation, params=None):
    # Load image
    img = load_image(img_path)
    img_height, img_width = get_image_size(img)

    # Randomly choose manipulation ROI
    roi_size = random.choice([512, 256, 128, 64])
    x = random.randint(0, img_width - roi_size)
    y = random.randint(0, img_height - roi_size)

    roi = (y, x, roi_size)

    # Ground truth image
    ground_truth = img.copy().astype(float)
    ground_truth[y:y + roi_size, x:x + roi_size] = np.inf
    ground_truth = np.where(ground_truth == np.inf, 255, 0).astype(np.uint8)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    # Apply manipulation
    # Copy-move
    if manipulation == 'copy_move':
        manip_img = copy_move(img_path, roi, img_width, img_height)  # TODO

    # Median filter
    elif manipulation == 'median_filter':
        if isinstance(params, int) and params > 0 and params % 2 != 0:  # Filter size must be an odd integer, otherwise use default value
            manip_img = median_filter(img, roi, params)
        else:
            manip_img = median_filter(img, roi)

    # Rotation
    elif manipulation == 'rotation':
        manip_img = rotate(img, roi) # TODO

    # Content-aware fill
    elif manipulation == 'content_aware_fill':
        if params == cv2.INPAINT_TELEA or params == cv2.INPAINT_NS:  # Method must be a valid OpenCV inpainting method
            manip_img = content_aware_fill(img, ground_truth, params)
        else:
            manip_img = content_aware_fill(img, ground_truth)



    #cv2.namedWindow('prova', cv2.WINDOW_NORMAL)
    #cv2.imshow('prova', manip_img)
    #cv2.waitKey(0)

    # TODO save image

    return manip_img, ground_truth


# Copy-move
def copy_move(img, roi, img_width, img_height):
    # Load image
    img = Image.open(img)

    # Define ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Randomly choose ROI to be copied
    x_copy = random.randint(0, img_width - roi_size)
    y_copy = random.randint(0, img_height - roi_size)

    # Manipulate ROI
    manip_roi = img.crop((x_copy, y_copy, x_copy + roi_size, y_copy + roi_size))

    # Generate manipulated image
    manip_img = img.copy()
    manip_img.paste(manip_roi, (x, y, x + roi_size, y + roi_size))

    return manip_img


# Median filter
def median_filter(img, roi, filter_size=3):
    # Define ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Manipulate ROI
    manip_roi = cv2.medianBlur(img[y:y + roi_size, x:x + roi_size], filter_size)

    # Generate manipulated image
    manip_img = img.copy()
    manip_img[y:y+roi_size, x:x+roi_size] = manip_roi

    return manip_img


# Rotation
def rotate(img, roi):
    # TODO
    return


# Content-aware fill
def content_aware_fill(img, ground_truth, method=cv2.INPAINT_TELEA):
    # Generate manipulated image
    manip_img = img.copy()
    manip_img = cv2.inpaint(manip_img, ground_truth, 3, flags=method)

    return manip_img


# TODO
manipulate_image('img/test.png', 'copy_move')
