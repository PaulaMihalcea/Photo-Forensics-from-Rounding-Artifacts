import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from decimal import Decimal
from preprocessing import average_block_from_window, get_windows, luminance, mfr
from em import expectation_maximization
from utils import get_filename


def main(args):

    img = cv2.imread(args.img_path)

    if img is None:  # Check that image has been loaded correctly
        raise IOError('Image loading error.')
    else:
        # RGB to YCbCr conversion & luminance channel extraction
        lum = luminance(img)

        # 3x3 median filter residual
        filtered_lum = mfr(lum, 3)

        # Overlapping windows generation
        windows = get_windows(filtered_lum, args.win_size, 8)

        # Average 8x8 block generation
        blocks = np.zeros((windows.shape[0], 8, 8))
        for i in range(windows.shape[0]):
            blocks[i] = average_block_from_window(windows[i], 8, 8)

        # Expectation-maximization algorithm
        prob_b_in_c1_r, c, diff_log = expectation_maximization(blocks, args.stop_threshold)

        # Output map
        # TODO

        # Plot difference
        plt.plot(diff_log)
        plt.xlabel('EM iteration')
        plt.ylabel('Average of the difference matrix between successive estimates of c')
        plt.show()  # TODO Comment out

        # Save results
        if args.save_result:
            # Create subfolder for current image & configuration
            filename, extension = get_filename(args.img_path)
            res_path = 'results/' + filename + '_' + extension + '_' + str(args.win_size) + '_' + '{:.0e}'.format(Decimal(args.stop_threshold))

            try:
                os.makedirs(res_path)
            except FileExistsError:  # Directory already exists
                pass

            # Save files
            # TODO save more results (output map & co.)
            plt.savefig(res_path + '/c_diff_plot.png')






if __name__ == '__main__':

    parser = ArgumentParser(description='Main script for the "Photo Forensics from Rounding Artifacts" project.')

    parser.add_argument('img_path', help='Path of the image to be analyzed.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 128 px).')
    parser.add_argument('-st', '--stop_threshold', type=float, help='Expectation-maximization algorithm stop threshold (default: 1e-5).')
    parser.add_argument('-sv', '--save_result', help='Save the result in the \'results\' folder (default: False).')

    args = parser.parse_args()

    if args.win_size is None:
        args.win_size = 128
    if args.stop_threshold is None:
        args.stop_threshold = 1e-2  # TODO also try 1e-3
    if args.save_result is None:
        args.save_result = False
    else:
        args.save_result = True

    main(args)
