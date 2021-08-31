import cv2
import numpy as np


# Load image
def load_image(img):
    # Load image
    if isinstance(img, str):  # If input is a file path, load it as an image
        img = cv2.imread(img)

    return img


# Adjust image size (uses padding; for square block partitioning)
def adjust_size(img, win_size, stride):

    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        return  # TODO Raise error

    if img_h % win_size != 0:
        img = cv2.copyMakeBorder(img, 0, (img_h - win_size) % stride, 0, 0, cv2.BORDER_REPLICATE)

    if img_w % win_size != 0:
        img = cv2.copyMakeBorder(img, 0, 0, 0, (img_w - win_size) % stride, cv2.BORDER_REPLICATE)

    return img


# RGB to YCbCr conversion & luminance channel extraction
def luminance(img):
    # Convert image
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, _, _ = cv2.split(img_y)

    return y


# Median filter residual
def mfr(img, size):
    # Ref.: https://www.researchgate.net/figure/Median-filter-residual_fig1_341836414

    median = cv2.medianBlur(img, size)
    residual = img - median

    return residual


# Blocks of given size (extracted with given stride)
def get_windows(img, win_size, stride):

    adjust_size(img, win_size, stride)

    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        return  # TODO Raise error

    x = []
    y = []
    blocks = []

    for i in range(0, img_w - stride, stride):
        x.append(i)

    for j in range(0, img_h - stride, stride):
        y.append(j)

    for i in y:
        for j in x:
            split = img[i:i + win_size, j:j + win_size]
            blocks.append(split)

    blocks = np.array(blocks)

    return blocks


# Average 8x8 block from given window
def average_block_from_window(window, block_size, block_stride):
    win_blocks = get_windows(window, block_size, block_stride)

    sum_block = np.sum(win_blocks, axis=0)

    avg_block = sum_block / len(win_blocks)

    return avg_block
