import time
import tqdm
from argparse import ArgumentParser
from preprocessing import get_average_window_blocks, luminance, mfr
from em import expectation_maximization
from postprocessing import get_output_map, get_roc_auc, get_template_difference_plot, plot_roc
from utils import load_image


def main(args):

    # Welcome message
    print('Photo Forensics from Rounding Artifacts: Python implementation')
    print('Author: Paula Mihalcea')
    print('Version: 1.0')
    print('Based on a research by S. Agarwal and H. Farid. Details & source code at https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts.')
    print()

    progress_bar = tqdm.tqdm(total=100)
    start = time.time()

    # Load image
    img_name = args.img_path.split('/')[-1]
    progress_bar.set_description('Loading image {}'.format(img_name))
    img = load_image(args.img_path)
    progress_bar.update(2)

    # RGB to YCbCr conversion & luminance channel extraction
    progress_bar.set_description('Extracting luminance')
    lum = luminance(img)
    progress_bar.update(3)

    # 3x3 median filter residual
    progress_bar.set_description('Getting median filter residual')
    filtered_lum = mfr(lum, 3)
    progress_bar.update(5)

    # Average blocks from overlapping windows generation
    progress_bar.set_description('Averaging blocks from overlapping windows')
    blocks, blocks_map = get_average_window_blocks(filtered_lum, args.win_size, 8, 8)
    progress_bar.update(20)

    # Expectation-maximization algorithm
    progress_bar.set_description('Executing EM algorithm')
    prob_b_in_c1_r, c, diff_history = expectation_maximization(blocks, args.stop_threshold, args.prob_r_b_in_c1)
    progress_bar.update(30)

    # Output map & difference plot
    progress_bar.set_description('Generating output map')
    output_map = get_output_map(prob_b_in_c1_r, blocks_map, img.shape[1], img.shape[0], args.show, args.save, args.img_path, args.win_size, args.stop_threshold, args.interpolate)
    get_template_difference_plot(diff_history, args.show_diff_plot, args.save_diff_plot, args.img_path, args.win_size, args.stop_threshold)
    progress_bar.update(35)

    # Compute ROC curve and AUC score
    progress_bar.set_description('Computing AUC score')
    fpr, tpr, auc = get_roc_auc(args.img_path, output_map)
    if auc is not None:
        plot_roc(fpr, tpr, auc, args.show_roc_plot, args.save_roc_plot, args.img_path, args.win_size, args.stop_threshold)
    progress_bar.update(5)

    end = time.time()

    # Final message
    print()
    print('Summary')
    print('Filename: {}.'.format(img_name))
    print('Image size: ' + str(img.shape[1]) + 'x' + str(img.shape[0]) + ' px.')
    if auc is not None:
        print('AUC score: {:.2f}.'.format(auc))
    else:
        print('No ground truth image found; AUC score unavailable.')
    if (end - start) / 60 >= 1:
        print('Elapsed time: {:.0f} m'.format((end - start) / 60) + ' {:.2f} s.'.format((end - start) % 60))
    else:
        print('Elapsed time: {:.2f} s.'.format(end - start))

    return output_map, auc


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Main script for the "Photo Forensics from Rounding Artifacts" project. Detects manipulated areas in JPEG images containing rounding artifacts as described in the referenced paper.')

    # Add parser arguments
    parser.add_argument('img_path', help='Path of the image to be analyzed.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 256). Note: must be a multiple of 8.')
    parser.add_argument('-st', '--stop_threshold', type=float, help='Expectation-maximization algorithm stop threshold (default: 1e-3).')
    parser.add_argument('-prob', '--prob_r_b_in_c1', type=float, help='Expectation-maximization algorithm probability of r conditioned by b belonging to C_1 (default: 0.5).')
    parser.add_argument('-int', '--interpolate', type=float, help='Interpolate missing pixel values, aka NaNs generated from divisions in the EM algorithm (default: False). Warning: slows down the program significantly.')
    parser.add_argument('-sh', '--show', help='Show the resulting output map (default: False).')
    parser.add_argument('-sv', '--save', help='Save the resulting output map in the \'results\' folder (default: False).')
    parser.add_argument('-shroc', '--show_roc_plot', help='Show the plot of the ROC curve (default: False).')
    parser.add_argument('-svroc', '--save_roc_plot', help='Save the plot of the ROC curve in the \'results\' folder (default: False).')
    parser.add_argument('-shdiff', '--show_diff_plot', help='Show the plot of the difference between successive estimates of template c (default: False).')
    parser.add_argument('-svdiff', '--save_diff_plot', help='Save the plot of the difference between successive estimates of template c in the \'results\' folder (default: False).')

    args = parser.parse_args()

    # Set default arguments
    if args.win_size is None:
        args.win_size = 256

    if args.stop_threshold is None:
        args.stop_threshold = 1e-3

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

    if args.show_roc_plot == 'True':
        args.show_roc_plot = True
    else:
        args.show_roc_plot = False

    if args.save_roc_plot == 'True':
        args.save_roc_plot = True
    else:
        args.save_roc_plot = False

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
