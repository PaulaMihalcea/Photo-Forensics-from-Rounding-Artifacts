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

    img_h, img_w = img.shape

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
    residual = median - img

    return residual


# Blocks of given size (extracted with given stride)
def get_blocks(img, block_size, stride):

    adjust_size(img, block_size, stride)

    img_h, img_w = img.shape

    x = []
    y = []
    blocks = []

    for i in range(0, img_w - block_size, stride):
        x.append(i)

    for j in range(0, img_h - block_size, stride):
        y.append(j)

    for i in y:
        for j in x:
            split = img[i:i + block_size, j:j + block_size]
            blocks.append(split)

    blocks = np.array(blocks)

    return blocks


# Average 8x8 block from given window
def average_block_from_window(window, block_size, block_stride):
    win_blocks = get_blocks(window, block_size, block_stride)

    sum_block = np.sum(win_blocks, axis=0)

    avg_block = sum_block / len(win_blocks)

    return avg_block
