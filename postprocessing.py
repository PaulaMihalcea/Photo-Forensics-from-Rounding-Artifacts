import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from interpolate_missing_pixels import interpolate_missing_pixels


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


# Get output map
def get_output_map(prob_b_in_c1_r, blocks_map, img_w, img_h, show=False, save=False, img_path=None, win_size=None, stop_threshold=None, interpolate=False):

    # Initialize empty map
    output_map = np.empty((img_h, img_w, 2))

    for w in blocks_map:  # For each element in the window list...
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 0] += prob_b_in_c1_r[w[2]]
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 1] += 1

    for i in range(0, output_map.shape[0]):  # Average
        for j in range(0, output_map.shape[1]):
            output_map[i, j, 0] = output_map[i, j, 0] / output_map[i, j, 1]

    output_map = 1 - output_map[:, :, 0]  # Because the map computed so far actually shows the probability that a pixel has not been modified

    # Replace NaNs using interpolation
    if interpolate:
        output_mask = np.ma.masked_invalid(output_map).mask
        output_map = interpolate_missing_pixels(output_map, output_mask, 'linear')

    # Matplotlib output map plot for debug purposes only
    # plt.imshow(output_map)
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.show()

    # Thresholding & normalization
    output_map_norm = np.where(output_map > 0.8, 1, 0).astype(np.uint8)
    output_map_norm = cv2.normalize(output_map_norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Normalize output map before saving

    # Show output map and/or save it to disk if requested
    filename, extension = get_filename(img_path)
    if show:
        cv2.namedWindow(filename + '.' + extension + ' output map', cv2.WINDOW_NORMAL)
        cv2.imshow(filename + '.' + extension + ' output map', output_map_norm)
        cv2.waitKey(0)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        cv2.imwrite(res_path + '/' + filename + '.png', output_map_norm)

    return output_map


# Plot difference between successive estimates of template c
def get_template_difference_plot(diff_history, show=False, save=False, img_path=None, win_size=None, stop_threshold=None):

    # Create plot
    plt.plot(diff_history)
    plt.xlabel('EM iteration')
    plt.xticks(range(0, len(diff_history)))
    plt.ylabel('Average of the difference matrix between successive estimates of c')

    # Save plot to disk (if requested, otherwise just show it)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        plt.savefig(res_path + '/c_diff_plot.png')
    if show:
        plt.show()

    return
