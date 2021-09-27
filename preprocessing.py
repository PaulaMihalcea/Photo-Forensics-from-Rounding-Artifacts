import cv2
import numpy as np
from utils import get_image_size


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


# Adjust image size (uses padding; for square window partitioning)
def adjust_size(img, win_size, stride):
    # Check image size and add padding if needed
    img_h, img_w = get_image_size(img)

    if img_h % stride != 0 or img_h % win_size != 0:
        img = cv2.copyMakeBorder(img, 0, stride - img_h % stride, 0, 0, cv2.BORDER_REPLICATE)

    if img_w % stride != 0 or img_w % win_size != 0:
        img = cv2.copyMakeBorder(img, 0, 0, 0, stride - img_w % stride, cv2.BORDER_REPLICATE)

    return img


# Calculate overlapping windows of given size (extracted with given stride; non-overlapping if stride is zero)
# and return their average block
# along with their starting coordinates and ID in order to map them to each pixel (for postprocessing)
def get_average_window_blocks(img, win_size, stride, block_size):
    # Adjust image size
    img = adjust_size(img, win_size, stride)
    img_h, img_w = get_image_size(img)

    # Calculate starting coordinates for each window (top left pixel)
    x = []
    y = []

    for i in range(0, img_w - win_size + 1, stride):
        x.append(i)

    for j in range(0, img_h - win_size + 1, stride):
        y.append(j)

    # Variable initialization
    window_id = 0  # Simple index to keep track of the current window
    blocks = np.zeros((len(x)*len(y), block_size, block_size))  # Final array of blocks (there is one block per window)
    blocks_map = np.zeros((len(x)*len(y), 3), dtype=np.int)  # Each row of blocks_map contains the coordinates of the top left pixel of the window (columns 0 and 1) and its ID (column 2)

    # Window average and mapping
    for i in y:
        for j in x:
            # Get current window blocks
            current_window_blocks = get_non_overlapping_blocks(img[i:i + win_size, j:j + win_size], block_size)

            # Calculate average block for the current window
            sum_block = np.sum(current_window_blocks, axis=0)
            blocks[window_id] = sum_block / len(current_window_blocks)

            # Update blocks map
            blocks_map[window_id, 0] = j
            blocks_map[window_id, 1] = i
            blocks_map[window_id, 2] = window_id

            # Update ID
            window_id += 1

    return blocks, blocks_map


# Non-overlapping blocks of given size
def get_non_overlapping_blocks(window, block_size):
    # Calculate starting coordinates for each window (top left pixel)
    x = []
    y = []

    for i in range(0, window.shape[0] - block_size, block_size):
        x.append(i)
        y.append(i)

    # Variable initialization
    window_id = 0
    blocks = np.zeros((len(x)*len(y), block_size, block_size))

    # Calculate blocks
    for i in y:
        for j in x:
            blocks[window_id] = window[i:i + block_size, j:j + block_size]
            window_id += 1

    return blocks
