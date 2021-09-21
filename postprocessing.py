import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.ndimage import gaussian_filter1d


# Get filename from path
def get_filename(file_path):
    tmp_filename = file_path.split('/')[-1]
    extension = tmp_filename.split('.')[-1]
    tmp_filename = tmp_filename.split('.')[:-1]

    if len(tmp_filename) > 1:
        filename = tmp_filename[0]
        for el in tmp_filename[1:]:
            filename += '.' + el
    else:
        filename = tmp_filename[0]

    return filename, extension


# Create results subfolder for current image & configuration & returns its path
def get_subfolder(img_path, win_size, stop_threshold):
    # Create subfolder name
    filename, extension = get_filename(img_path)
    res_path = 'results/' + filename + '_' + extension + '_' + str(win_size) + '_' + '{:.0e}'.format(Decimal(stop_threshold))

    # Create subfolder
    try:
        os.makedirs(res_path)
    except FileExistsError:  # Directory already exists
        pass

    return res_path


# Interpolate missing pixels (NaNs)
# Function by StackOverflow user Sam De Meyer, based on user G M's answer:
# https://stackoverflow.com/a/68558547
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


# Get output map
def get_output_map(prob_b_in_c1_r, blocks_map, img_w, img_h, save=False, img_path=None, win_size=None, stop_threshold=None):

    # Initialize empty map
    output_map = np.empty((img_w, img_h))  # TODO check

    # Start looping through original image pixels
    for i in range(0, img_w):
        for j in range(0, img_h):

            # Variable initialization
            current_probability = 0  # Simple counter; keeps track of the probability being processed
            current_pixel_probabilities = np.empty(len(blocks_map[i][j]))  # Contains all probabilities assigned to each block containing the current pixel (blocks_map[j][i])

            # Probability extraction from prob_b_in_c1_r vector
            for w in blocks_map[i][j]:  # w is the identifier of each window containing the current pixel
                current_pixel_probabilities[current_probability] = prob_b_in_c1_r[w]
                current_probability += 1

            # Average probability per pixel
            output_map[i][j] = np.average(current_pixel_probabilities)

    # Replace NaNs using interpolation
    output_mask = np.ma.masked_invalid(output_map).mask
    output_map = interpolate_missing_pixels(output_map, output_mask, 'linear')

    #'''
    plt.imshow(1-output_map)  # TODO duplicate plot; cv2 should be deleted and this function used instead with a greyscale colormap
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()
    #'''

    # Save output map to disk (if requested, otherwise just show it)
    filename, extension = get_filename(img_path)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        cv2.imwrite(res_path + '/' + filename + '.png', output_map)
    else:
        cv2.imshow(filename + '.' + extension + ' output map', output_map)
        cv2.waitKey(0)

    return output_map


# Plot difference between successive estimates of template c
def get_template_difference_plot(diff_history, save=False, img_path=None, win_size=None, stop_threshold=None):

    # Create plot
    plt.plot(diff_history)
    plt.xlabel('EM iteration')
    plt.xticks(range(0, len(diff_history)))
    plt.ylabel('Average of the difference matrix between successive estimates of c')

    # Save plot to disk (if requested, otherwise just show it)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        plt.savefig(res_path + '/c_diff_plot.png')
    else:
        plt.show()

    return
