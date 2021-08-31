import cv2
import numpy as np
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# Load image
def load_image(img):
    # Load image
    if isinstance(img, str):  # If input is a file path, load it as an image
        img = cv2.imread(img)

    return img


# Adjust image size (uses padding; for square block partitioning)
#@jit(nopython=False)
def adjust_size(img, win_size, stride):

    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        return  # TODO Raise error

    if img_h % stride != 0 or img_h % win_size != 0:
        #img = cv2.copyMakeBorder(img, 0, (img_h - win_size) % stride, 0, 0, cv2.BORDER_REPLICATE)  # TODO
        img = cv2.copyMakeBorder(img, 0, stride - img_h % stride, 0, 0, cv2.BORDER_REPLICATE)

    if img_w % stride != 0 or img_w % win_size != 0:
        #img = cv2.copyMakeBorder(img, 0, 0, 0, (img_w - win_size) % stride, cv2.BORDER_REPLICATE)
        img = cv2.copyMakeBorder(img, 0, 0, 0, stride - img_w % stride, cv2.BORDER_REPLICATE)

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
#@jit(nopython=False, forceobj=False)
def get_windows_new(img, win_size, stride, block_size):

    img = adjust_size(img, win_size, stride)

    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        return  # TODO Raise error

    x = []
    y = []

    for i in range(0, img_w - win_size, stride):
        x.append(i)

    for j in range(0, img_h - win_size, stride):
        y.append(j)


    counter = 0

    blocks = np.zeros((len(x)*len(y), block_size, block_size))

    for i in y:
        for j in x:
            win_blocks = get_blocks(img[i:i + win_size, j:j + win_size], block_size)

            sum_block = np.sum(win_blocks, axis=0)

            blocks[counter] = sum_block / len(win_blocks)
            #print('done window ' + str(counter) + '/' + total)
            counter += 1

    #blocks = np.array(blocks)

    return blocks


# TODO
#@jit(nopython=False, forceobj=False)
def get_blocks(window, block_size):
    x = []
    y = []

    for i in range(0, window.shape[0] - block_size, block_size):
        x.append(i)
        y.append(i)

    blocks = np.zeros((len(x)*len(y), block_size, block_size))
    counter = 0

    for i in y:
        for j in x:
            blocks[counter] = window[i:i + block_size, j:j + block_size]
            counter += 1

    #blocks = np.array(blocks, dtype='object')

    return blocks


# Average 8x8 block from given window
#@jit(nopython=False)
def average_block_from_window(window, block_size, block_stride):
    #win_blocks = get_windows(window, block_size, block_stride, True)
    win_blocks = get_blocks(window, block_size)

    sum_block = np.sum(win_blocks, axis=0)

    avg_block = sum_block / len(win_blocks)

    return avg_block
