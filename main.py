from argparse import ArgumentParser
from preprocessing import get_windows, load_image, luminance, mfr
from em import expectation_maximization
from postprocessing import get_template_difference_plot, get_output_map


def main(args):

    # Load image
    print('Loading image... ', end='')
    img = load_image(args.img_path)
    print('done.')

    # RGB to YCbCr conversion & luminance channel extraction
    print('Luminance extraction... ', end='')
    lum = luminance(img)
    print('done.')

    # 3x3 median filter residual
    print('Getting median filter residual... ', end='')
    filtered_lum = mfr(lum, 3)
    print('done.')

    # Average blocks from overlapping windows generation
    print('Averaging blocks from overlapping windows... ', end='')
    blocks, blocks_map = get_windows(filtered_lum, args.win_size, 8, 8)
    print('done.')

    # Expectation-maximization algorithm
    print('Executing EM algorithm... ', end='')
    prob_b_in_c1_r, c, diff_history = expectation_maximization(blocks, args.stop_threshold)
    print('done.')

    # Output map & difference plot
    print('Generating output map... ', end='')
    output_map = get_output_map(prob_b_in_c1_r, blocks_map, img.shape[1], img.shape[0], args.save, args.img_path, args.win_size, args.stop_threshold)
    get_template_difference_plot(diff_history, args.save, args.img_path, args.win_size, args.stop_threshold)
    print('done.')

    return


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Main script for the "Photo Forensics from Rounding Artifacts" project.')

    # Add parser arguments
    parser.add_argument('img_path', help='Path of the image to be analyzed.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 128 px).')
    parser.add_argument('-st', '--stop_threshold', type=float, help='Expectation-maximization algorithm stop threshold (default: 1e-3).')
    parser.add_argument('-sv', '--save', help='Save the results in the \'results\' folder (default: False).')

    args = parser.parse_args()

    # Set default arguments
    if args.win_size is None:
        args.win_size = 256
    if args.stop_threshold is None:
        args.stop_threshold = 1e-3  # TODO Try 1e-2/1e-3
    if args.save == 'True':
        args.save = True
    else:
        args.save = False

    # Run main script
    main(args)
