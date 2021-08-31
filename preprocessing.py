import cv2
import numpy as np


# Load image
def load_image(img):
    # Load image
    img = cv2.imread(img)

    # Check image correctness
    if img is not None:
        return img
    else:
        raise IOError('Error while loading image: invalid image file or image file path.')


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


# Get image size (regardless of number of channels)
def get_image_size(img):
    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        raise RuntimeError('Incorrect input image shape.')

    return img_w, img_h


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
def get_windows(img, win_size, stride, block_size):
    # Get image size
    img_h_orig, img_w_orig = get_image_size(img)

    # Initialize blocks map
    # Each point of the map corresponds to a pixel, and contains a list containing the IDs of the windows containing that pixel
    blocks_map = np.zeros((img_h_orig, img_w_orig, 0)).tolist()

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

    # Variables initialization
    window_id = 0
    blocks = np.zeros((len(x)*len(y), block_size, block_size))  # Final array of blocks (there is one block per window)

    for i in y:
        for j in x:
            # Get current window blocks
            current_window_blocks = get_blocks(img[i:i + win_size, j:j + win_size], block_size)

            # Update blocks map
            for m in range(i, i + win_size):
                if m < img_h_orig:
                    for n in range(j, j + win_size):
                        if n < img_w_orig:
                            blocks_map[m][n].append(window_id)

            # Calculate average block for the current window
            sum_block = np.sum(current_window_blocks, axis=0)
            blocks[window_id] = sum_block / len(current_window_blocks)

            # Update ID
            window_id += 1

    return blocks, blocks_map


# Non-overlapping blocks of given size
def get_blocks(window, block_size):
    # Calculate starting coordinates for each window (top left pixel)
    x = []
    y = []

    for i in range(0, window.shape[0] - block_size, block_size):
        x.append(i)
        y.append(i)

    # Variables initialization
    window_id = 0
    blocks = np.zeros((len(x)*len(y), block_size, block_size))

    # Calculate blocks
    for i in y:
        for j in x:
            blocks[window_id] = window[i:i + block_size, j:j + block_size]
            window_id += 1

    return blocks
