import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tqdm
from argparse import Namespace
from main import main
from manipulate import get_manip_id
from postprocessing import plot_roc
from utils import get_filename, get_image_info


# Images path
jpeg_dir_path = 'img/manip_jpeg/'
png_dir_path = 'img/manip_png/'

# Save/load data
save = False

results_jpeg_path = 'results_jpeg.npy'
fpr_jpeg_path = 'fpr_jpeg.npy'
tpr_jpeg_path = 'tpr_jpeg.npy'
results_png_path = 'results_png.npy'
fpr_png_path = 'fpr_png.npy'
tpr_png_path = 'tpr_png.npy'

# File lists
valid_jpeg_extensions = ['.jpeg', '.jpg', '.jpe', '.jfif', '.jif']
valid_png_extensions = ['.png']
jpeg_file_list = [jpeg_dir_path + img_file for img_file in os.listdir(jpeg_dir_path) for ext in valid_jpeg_extensions if img_file.endswith(ext)]
png_file_list = [png_dir_path + img_file for img_file in os.listdir(png_dir_path) for ext in valid_png_extensions if img_file.endswith(ext)]

# Compute and save results
if save:
    # Setup; uses default parameters
    args = Namespace()
    args.win_size = 256
    args.stop_threshold = 1e-2
    args.prob_r_b_in_c1 = 0.3
    args.interpolate = False
    args.show = False
    args.save = False
    args.show_roc_plot = False
    args.save_roc_plot = False
    args.show_diff_plot = False
    args.save_diff_plot = False

    print('Results script')
    print()

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
    #sys.stdout = open(os.devnull, 'w')  # Suppress calls to print()  # TODO

    # JPEG images
    for i in jpeg_file_list:
        # Image information
        args.img_path = i
        filename, extension = get_filename(i)
        win_size, manip, quality = get_image_info(filename, '.' + extension)

        # Update progress bar
        progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

        # Main EM algorithm
        output_map, auc, fpr, tpr = main(args)

        # Append results
        #results_jpeg.append([win_size, get_manip_id(manip), quality, auc])  # TODO
        results_jpeg.append([win_size, get_manip_id(manip), quality, auc, fpr, tpr])
        # TODO
        '''
        if len(fpr_jpeg) == 0:
            fpr_jpeg = fpr
        else:
            if len(fpr_jpeg) < len(fpr):  # Make sure that arrays are the same length
                fpr_sum = np.zeros(len(fpr))
                fpr_sum[:len(fpr_jpeg)] += fpr_jpeg
            else:
                fpr_sum = np.zeros(len(fpr_jpeg))
                fpr_sum[:len(fpr)] += fpr
            fpr_jpeg = fpr_sum
            if len(tpr_jpeg) == 0:
                tpr_jpeg = tpr
            else:
                if len(tpr_jpeg) < len(tpr):  # Make sure that arrays are the same length
                    tpr_sum = np.zeros(len(tpr))
                    tpr_sum[:len(tpr_jpeg)] += tpr_jpeg
                else:
                    tpr_sum = np.zeros(len(tpr_jpeg))
                    tpr_sum[:len(fpr)] += tpr
                tpr_jpeg = tpr_sum
        '''

        # Update progress bar (again)
        progress_bar.update(1)

    # PNG images
    for i in png_file_list:
        # Image information
        args.img_path = i
        filename, extension = get_filename(i)
        win_size, manip, _ = get_image_info(filename, '.' + extension)

        # Update progress bar
        progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

        # Main EM algorithm
        output_map, auc, fpr, tpr = main(args)

        # Append results
        #results_png.append([win_size, get_manip_id(manip), -1, auc])  # TODO
        results_png.append([win_size, get_manip_id(manip), -1, auc, fpr, tpr])
        # TODO
        '''
        if len(fpr_png) == 0:
            fpr_png = fpr
        else:
            if len(fpr_png) < len(fpr):  # Make sure that arrays are the same length
                fpr_sum = np.zeros(len(fpr))
                fpr_sum[:len(fpr_png)] += fpr_png
            else:
                fpr_sum = np.zeros(len(fpr_png))
                fpr_sum[:len(fpr)] += fpr
            fpr_png = fpr_sum
            if len(tpr_png) == 0:
                tpr_png = tpr
            else:
                if len(tpr_png) < len(tpr):  # Make sure that arrays are the same length
                    tpr_sum = np.zeros(len(tpr))
                    tpr_sum[:len(tpr_png)] += tpr_png
                else:
                    tpr_sum = np.zeros(len(tpr_png))
                    tpr_sum[:len(fpr)] += tpr
                tpr_png = tpr_sum
        '''

        # Update progress bar (again)
        progress_bar.update(1)

    sys.stdout = sys.__stdout__  # Enable calls to print()

    # Results as NumPy arrays
    results_jpeg = np.asarray(results_jpeg)
    fpr_jpeg = np.asarray(fpr_jpeg)
    tpr_jpeg = np.asarray(tpr_jpeg)
    results_png = np.asarray(results_png)
    fpr_png = np.asarray(fpr_png)
    tpr_png = np.asarray(tpr_png)

    # Save results
    if not os.path.exists('results/'):
        os.makedirs('results/')

    np.save('results/' + results_jpeg_path, results_jpeg)
    np.save('results/' + fpr_jpeg_path, fpr_jpeg)
    np.save('results/' + tpr_jpeg_path, tpr_jpeg)
    np.save('results/' + results_png_path, results_png)
    np.save('results/' + fpr_png_path, fpr_png)
    np.save('results/' + tpr_png_path, tpr_png)

# Load results
else:
    results_jpeg = np.load('results/' + results_jpeg_path)
    fpr_jpeg = np.load('results/' + fpr_jpeg_path)
    tpr_jpeg = np.load('results/' + tpr_jpeg_path)
    results_png = np.load('results/' + results_png_path)
    fpr_png = np.load('results/' + fpr_png_path)
    tpr_png = np.load('results/' + tpr_png_path)

# ROC curve for all images
mean_auc_512 = np.mean([np.mean([results_jpeg[results_jpeg[:, 0] == 512, 3], np.mean(results_png[results_png[:, 0] == 512, 3])])])
mean_auc_256 = np.mean([np.mean([results_jpeg[results_jpeg[:, 0] == 256, 3], np.mean(results_png[results_png[:, 0] == 256, 3])])])
mean_auc_128 = np.mean([np.mean([results_jpeg[results_jpeg[:, 0] == 128, 3], np.mean(results_png[results_png[:, 0] == 128, 3])])])
mean_auc_64 = np.mean([np.mean([results_jpeg[results_jpeg[:, 0] == 64, 3], np.mean(results_png[results_png[:, 0] == 64, 3])])])

fpr_jpeg = fpr_jpeg / len(jpeg_file_list)
tpr_jpeg = tpr_jpeg / len(jpeg_file_list)
fpr_png = fpr_png / len(png_file_list)
tpr_png = tpr_png / len(png_file_list)

# Make sure that arrays are the same length
if len(fpr_jpeg) < len(fpr_png):
    fpr_sum = np.zeros(len(fpr_png))
    fpr_sum[:len(fpr_jpeg)] += fpr_jpeg
else:
    fpr_sum = np.zeros(len(fpr_jpeg))
    fpr_sum[:len(fpr_png)] += fpr_png
if len(tpr_jpeg) < len(tpr_png):
    tpr_sum = np.zeros(len(tpr_png))
    tpr_sum[:len(tpr_jpeg)] += tpr_jpeg
else:
    tpr_sum = np.zeros(len(tpr_jpeg))
    tpr_sum[:len(tpr_png)] += tpr_png

print([mean_auc_512, mean_auc_256, mean_auc_128, mean_auc_64])

plot_roc(fpr_sum / 2, tpr_sum / 2, np.asarray([mean_auc_512, mean_auc_256, mean_auc_128, mean_auc_64]), show=False, save=True)
