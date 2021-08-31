import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


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
def get_output_map(prob_b_in_c1_r, blocks_map, img_w, img_h, save=False, img_path=None, win_size=None, stop_threshold=None):

    # Initialize empty map
    output_map = np.empty((img_h, img_w))

    # Start looping through original image pixels
    for i in range(0, img_w):
        for j in range(0, img_h):

            # Variable initialization
            current_probability = 0  # Simple counter; keeps track of the probability being processed
            current_pixel_probabilities = np.empty(len(blocks_map[j][i]))  # Contains all probabilities assigned to each block containing the current pixel (blocks_map[j][i])

            # Probability extraction from prob_b_in_c1_r vector
            for w in blocks_map[j][i]:  # w is the identifier of each window containing the current pixel
                current_pixel_probabilities[current_probability] = prob_b_in_c1_r[w]
                current_probability += 1

            # Average probability per pixel
            output_map[j][i] = np.average(current_pixel_probabilities)

    # Normalization
    output_map *= (255.0 / output_map.max())  # TODO Check this formula's correctness

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
