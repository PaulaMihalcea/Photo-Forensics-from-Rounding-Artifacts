import cv2
import numpy as np
from argparse import ArgumentParser
from preprocessing import average_block_from_window, get_blocks, luminance, mfr
from em import expectation_maximization


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
        windows = get_blocks(filtered_lum, args.win_size, 8)

        # Average 8x8 block generation
        blocks = np.zeros((windows.shape[0], 8, 8))
        for i in range(windows.shape[0]):
            blocks[i] = average_block_from_window(windows[i], 8, 8)

        # Expectation-maximization algorithm
        prob_b_in_c1_r, c = expectation_maximization(blocks, args.stop_threshold)

        # Output map
        # TODO





if __name__ == '__main__':

    parser = ArgumentParser(description='Main script for the "Photo Forensics from Rounding Artifacts" project.')

    parser.add_argument('img_path', help='Path of the image to be analyzed.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 128 px).')
    parser.add_argument('-st', '--stop_threshold', type=float, help='Expectation-maximization algorithm stop threshold (default: 1e-5).')

    args = parser.parse_args()

    if args.win_size is None:
        args.win_size = 128
    if args.stop_threshold is None:
        args.stop_threshold = 1e-5

    main(args)
