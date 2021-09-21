import time
from argparse import ArgumentParser
from preprocessing import get_windows, load_image, luminance, mfr
from em import expectation_maximization
from postprocessing import get_template_difference_plot, get_output_map


def main(args):

    start = time.time()

    # Load image
    img_name = args.img_path.split('/')[-1]
    print('Loading image ' + img_name + '... ', end='')
    img = load_image(args.img_path)
    print('done; image size: ' + str(img.shape[1]) + 'x' + str(img.shape[0]) + '.')

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
    prob_b_in_c1_r, c, diff_history = expectation_maximization(blocks, args.stop_threshold, args.prob_r_b_in_c1)
    print('done.')

    # Output map & difference plot
    print('Generating output map... ', end='')
    output_map = get_output_map(prob_b_in_c1_r, blocks_map, img.shape[1], img.shape[0], args.show, args.save, args.img_path, args.win_size, args.stop_threshold, args.interpolate)
    get_template_difference_plot(diff_history, args.show_diff_plot, args.save_diff_plot, args.img_path, args.win_size, args.stop_threshold)
    print('done.')
    end = time.time()

    print('Elapsed time: ' + str('{:.2f}'.format(end - start)) + ' s.')

    return


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Main script for the "Photo Forensics from Rounding Artifacts" project.')

    # Add parser arguments
    parser.add_argument('img_path', help='Path of the image to be analyzed.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 128 px).')
    parser.add_argument('-st', '--stop_threshold', type=float, help='Expectation-maximization algorithm stop threshold (default: 1e-3).')
    parser.add_argument('-st', '--prob_r_b_in_c1', type=float, help='Expectation-maximization algorithm probability of r conditioned by b belonging to C_1 (default: 0.5).')
    parser.add_argument('-int', '--interpolate', type=float, help='Interpolate missing pixel values, aka NaNs generated from divisions in the EM algorithm (default: False).')
    parser.add_argument('-sh', '--show', help='Show the resulting output map (default: False).')
    parser.add_argument('-sv', '--save', help='Save the resulting output map in the \'results\' folder (default: False).')
    parser.add_argument('-shdiff', '--show_diff_plot', help='Show the plot of the difference between successive estimates of template c (default: False).')
    parser.add_argument('-svdiff', '--save_diff_plot', help='Save the plot of the difference between successive estimates of template c in the \'results\' folder (default: False).')

    args = parser.parse_args()

    # Set default arguments
    if args.win_size is None:
        args.win_size = 256

    if args.stop_threshold is None:
        args.stop_threshold = 1e-3  # TODO Try 1e-2/1e-3

    if args.prob_r_b_in_c1 is None:
        args.prob_r_b_in_c1 = 0.5

    if args.interpolate == 'True':
        args.interpolate = True
    else:
        args.interpolate = False

    if args.show == 'True':
        args.show = True
    else:
        args.show = False

    if args.save == 'True':
        args.save = True
    else:
        args.save = False

    if args.show_diff_plot == 'True':
        args.show_diff_plot = True
    else:
        args.show_diff_plot = False

    if args.save_diff_plot == 'True':
        args.save_diff_plot = True
    else:
        args.save_diff_plot = False

    # Run main script
    main(args)
