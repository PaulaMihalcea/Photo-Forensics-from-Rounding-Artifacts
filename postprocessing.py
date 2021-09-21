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
def get_output_map(prob_b_in_c1_r, blocks_map, img_w, img_h, save=False, img_path=None, win_size=None, stop_threshold=None, interpolate=False):

    # Initialize empty map
    output_map = np.empty((img_h, img_w, 2))

    for w in blocks_map:  # For each element in the window list...
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 0] += prob_b_in_c1_r[w[2]]
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 1] += 1

    for i in range(0, output_map.shape[0]):  # Average
        for j in range(0, output_map.shape[1]):
            output_map[i, j, 0] = output_map[i, j, 0] / output_map[i, j, 1]

    output_map = output_map[:, :, 0]

    # Replace NaNs using interpolation
    if interpolate:
        output_mask = np.ma.masked_invalid(output_map).mask
        output_map = interpolate_missing_pixels(output_map, output_mask, 'linear')

    #'''
    plt.imshow(output_map)  # TODO duplicate plot; cv2 should be deleted and this function used instead with a greyscale colormap
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
        pass  # TODO
        #cv2.imshow(filename + '.' + extension + ' output map', output_map)
        #cv2.waitKey(0)

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
        pass
        #plt.show()# TODO

    return
