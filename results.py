import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import tqdm
from argparse import ArgumentParser, Namespace
from main import main as mm
from postprocessing import plot_roc#, get_mean_roc  # TODO
from utils import get_filename, get_jpeg_file_list, get_png_file_list


# Get image information from filename
def get_image_info(filename, extension):
    # Split filename
    el = filename.split('_')

    # Extract information
    if extension in ['png', 'PNG']:
        quality = None
        win_size = int(el[-1])
        manip = el[-2]
    elif extension in ['jpeg', 'jpg', 'jpe', 'jfif', 'jif', 'JPEG', 'JPG', 'JPE', 'JFIF', 'JIF']:
        quality = int(el[-1])
        win_size = int(el[-2])
        manip = el[-3]
    else:
        raise ValueError('Invalid file extension (allowed extensions: .png, .jpeg, .jpg, .jpe, .jfif or .jif.)')

    return win_size, manip, quality


# Manipulation ID
def get_manip_id(manip_name):
    if manip_name == 'copy-move':
        manip_id = 1
    elif manip_name == 'median-filter':
        manip_id = 2
    elif manip_name == 'rotation':
        manip_id = 3
    elif manip_name == 'content-aware-fill':
        manip_id = 4
    else:
        raise ValueError('Invalid manipulation method. Possible values: "copy-move", "median-filter", "rotation", "content-aware-fill".')

    return manip_id


# Manipulation name
def get_manip_name(manip_id):
    if manip_id == 1:
        manip_name = 'copy-move'
    elif manip_id == 2:
        manip_name = 'median-filter'
    elif manip_id == 3:
        manip_name = 'rotation'
    elif manip_id == 4:
        manip_name = 'content-aware-fill'
    else:
        raise ValueError('Invalid manipulation method. Possible values: 1, 2, 3, 4.')

    return manip_name


def main(args):
    # Welcome message
    print('Photo Forensics from Rounding Artifacts: results script')
    print('Author: Paula Mihalcea')
    print('Version: 1.0')
    print('Based on a research by S. Agarwal and H. Farid. Details & source code at https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts.')
    print()

    # Analyze images and save results
    if args.generate:

        # Get file list
        jpeg_file_list = get_jpeg_file_list(args.jpeg_dir_path)
        png_file_list = get_png_file_list(args.png_dir_path)

        # Create results subfolder
        if not os.path.exists('results/'):
            os.makedirs('results/')

        # Main setup (uses default parameters)
        args_mm = Namespace()
        args_mm.stop_threshold = 1e-2
        args_mm.prob_r_b_in_c1 = 0.3
        args_mm.interpolate = False
        args_mm.show = False
        args_mm.save = False
        args_mm.show_roc_plot = False
        args_mm.save_roc_plot = False
        args_mm.show_diff_plot = False
        args_mm.save_diff_plot = False
        args_mm.verbose = False

        win_sizes = [64, 128, 256]
        total_imgs = 0  # TODO update

        # Progress bar
        progress_bar = tqdm.tqdm(total=(len(jpeg_file_list) + len(png_file_list)) * len(win_sizes))

        # Time
        start = time.time()

        # Main loop for different window sizes
        for win_size in win_sizes:
            args_mm.win_size = win_size
            # First main loop: JPEG images
            for i in jpeg_file_list:
                # Image information
                args_mm.img_path = i
                filename, extension = get_filename(i)
                win_size, manip, quality = get_image_info(filename, extension)

                # Update progress bar
                progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format('.' + extension)))

                # Main EM algorithm
                output_map, auc, fpr, tpr = mm(args_mm)

                # Append results
                # TODO use pandas
                results_jpeg.append([win_size, get_manip_id(manip), quality, auc, fpr, tpr])

                # Update progress bar (again)
                progress_bar.update(1)

            # Second main loop: PNG images
            for i in png_file_list:
                # Image information
                args_mm.img_path = i
                filename, extension = get_filename(i)
                win_size, manip, _ = get_image_info(filename, extension)

                # Update progress bar
                progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

                # Main EM algorithm
                output_map, auc, fpr, tpr = mm(args_mm)

                # Append results
                # TODO use pandas
                results_png.append([win_size, get_manip_id(manip), -1, auc, fpr, tpr])

                # Update progress bar (again)
                progress_bar.update(1)

        # Save results
        # TODO save results

        end = time.time()

        # Final message
        print()
        print('Summary')
        print('Images analyzed: {}/'.format(total_imgs) + '{} (missing images have generated errors).'.format(len(jpeg_file_list) + len(png_file_list)) * len(win_sizes))
        if (end - start) / 60 ** 2 >= 1:
            print('Elapsed time: {:.0f} h'.format((end - start) / 60 ** 2) + ' {:.0f} m'.format((end - start) / 60) + ' {:.2f} s.'.format((end - start) % 60))
        elif (end - start) / 60 >= 1:
            print('Elapsed time: {:.0f} m'.format((end - start) / 60) + ' {:.2f} s.'.format((end - start) % 60))
        else:
            print('Elapsed time: {:.2f} s.'.format(end - start))

    else:
        # TODO load results from file
        pass

    # TODO generate plots from results

    return


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Results script for the "Photo Forensics from Rounding Artifacts" project; generates results by analyzing JPEG and PNG images from two given directories (e.g. "path/to/jpeg_images/" and "path/to/png_images/") as described in the referenced paper, or loads existing results from the "results/" folder.')

    # Add parser arguments
    parser.add_argument('generate', help='Analyze images and generate results; only loads existing results if False (default: False).')
    parser.add_argument('-jpeg', '--jpeg_dir_path', help='Path of the directory containing the JPEG images to be analyzed (only needed if generate is True).')
    parser.add_argument('-png', '--png_dir_path', help='Path of the directory containing the PNG images to be analyzed (only needed if generate is True).')

    args = parser.parse_args()

    # Run main script
    main(args)
