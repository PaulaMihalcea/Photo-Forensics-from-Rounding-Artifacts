import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tqdm
from argparse import Namespace
from main import main
from postprocessing import plot_roc
from utils import get_filename, get_image_info


# Setup; uses default parameters
args = Namespace()
args.win_size = 256
args.stop_threshold = 1e-2
args.interpolate = False
args.show = False
args.save = False
args.show_roc_plot = False
args.save_roc_plot = False
args.show_diff_plot = False
args.save_diff_plot = False

print('Results script...')
print()

# Images path
jpeg_dir_path = ''
png_dir_path = ''

# File list
valid_jpeg_extensions = ['.jpeg', '.jpg', '.jpe', '.jfif', '.jif']
valid_png_extensions = ['.png']
jpeg_file_list = [jpeg_dir_path + img_file for img_file in os.listdir(jpeg_dir_path) for ext in valid_jpeg_extensions if img_file.endswith(ext)]
png_file_list = [png_dir_path + img_file for img_file in os.listdir(png_dir_path) for ext in valid_png_extensions if img_file.endswith(ext)]

# Results
results_jpeg = []
fpr_jpeg = []
tpr_jpeg = []
results_png = []
fpr_png = []
tpr_png = []

# Progress bar
progress_bar = tqdm.tqdm(total=len(jpeg_file_list)*len(png_file_list))

# Main loop
sys.stdout = open(os.devnull, 'w')  # Suppress calls to print()

# JPEG images
for i in jpeg_file_list:
    # Image information
    filename, extension = get_filename(i)
    win_size, manip, quality = get_image_info(filename, extension)

    # Update progress bar
    progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

    # Main EM algorithm
    output_map, auc, fpr, tpr = main(args)

    # Append results
    results_jpeg.append([win_size, manip, quality, auc])

    # Update progress bar (again)
    progress_bar.update(1)

# PNG images
for i in png_file_list:
    # Image information
    filename, extension = get_filename(i)
    win_size, manip, _ = get_image_info(filename, extension)

    # Update progress bar
    progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

    # Main EM algorithm
    output_map, auc, fpr, tpr = main(args)

    # Append results
    results_png.append([win_size, manip, -1, auc])

    # Update progress bar (again)
    progress_bar.update(1)

sys.stdout = sys.__stdout__  # Enable calls to print()

# ROC curve for all images
if auc != 0:
    plot_roc(fpr, tpr, auc, args.show_roc_plot, args.save_roc_plot, args.img_path, args.win_size, args.stop_threshold)
