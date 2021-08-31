import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from utils import get_filename


def normalize():
    return


def get_output_map(prob_per_window, pixel_map, img_w, img_h):

    probs_map = np.empty((img_h, img_w))

    for i in range(0, img_w):
        for j in range(0, img_h):
            #print(pixel_map[i][j])
            probs = np.empty(len(pixel_map[i][j]))
            counter = 0
            for w in pixel_map[j][i]:  # w is the identifier of the window containing the current pixel
                probs[counter] = prob_per_window[w]
                #print(windows[w])
                counter += 1
                #print(windows[w])
            probs_map[j][i] = np.average(probs)

    #probs_map *= (255.0/probs_map.max()) # TODO non torna (?)
    # TODO va normalizzata, che son tutti valori di probabilità
    # TODO (o forse no?)
    print(probs_map)

    cv2.imshow('image', probs_map)
    cv2.waitKey(0)
    cv2.imwrite('results/test.png', probs_map)

            #new_map[i][j] =
            #win_blocks = get_blocks(img[i:i + win_size, j:j + win_size], block_size)

    return probs_map


def get_template_difference_plot(diff_log,):
    # Plot difference
    plt.plot(diff_log)
    plt.xlabel('EM iteration')
    plt.xticks(range(0, len(diff_log)))
    plt.ylabel('Average of the difference matrix between successive estimates of c')
    plt.show()

    return


def save_results(img_path, win_size, stop_threshold):
    # Create subfolder for current image & configuration
    filename, extension = get_filename(img_path)
    res_path = 'results/' + filename + '_' + extension + '_' + str(win_size) + '_' + '{:.0e}'.format(
        Decimal(stop_threshold))

    try:
        os.makedirs(res_path)
    except FileExistsError:  # Directory already exists
        pass

    # Save files
    # TODO save more results (output map & co.)
    plt.savefig(res_path + '/c_diff_plot.png')
    return
