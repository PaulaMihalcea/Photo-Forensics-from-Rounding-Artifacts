import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tqdm
from argparse import Namespace
from main import main
from manipulate import get_manip_id
from postprocessing import plot_roc, get_mean_roc
from utils import get_filename, get_image_info


# Images path
jpeg_dir_path = 'img/manip_jpeg/'
png_dir_path = 'img/manip_png/'

# Save/load data
save = True

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
    args.win_size = 64  # TODO
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
    progress_bar = tqdm.tqdm(total=len(jpeg_file_list)+len(png_file_list))

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
        results_jpeg.append([win_size, get_manip_id(manip), quality, auc, fpr, tpr])

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
        results_png.append([win_size, get_manip_id(manip), -1, auc, fpr, tpr])

        # Update progress bar (again)
        progress_bar.update(1)

    sys.stdout = sys.__stdout__  # Enable calls to print()

    # Results as NumPy arrays
    results_jpeg = np.asarray(results_jpeg)
    results_png = np.asarray(results_png)

    fprs_jpeg = np.stack(results_jpeg[:, 4])
    fprs_jpeg_512 = np.stack(results_jpeg[results_jpeg[:, 0] == 512, 4])
    fprs_jpeg_256 = np.stack(results_jpeg[results_jpeg[:, 0] == 256, 4])
    fprs_jpeg_128 = np.stack(results_jpeg[results_jpeg[:, 0] == 128, 4])
    fprs_jpeg_64 = np.stack(results_jpeg[results_jpeg[:, 0] == 64, 4])

    tprs_jpeg = np.stack(results_jpeg[:, 5])
    tprs_jpeg_512 = np.stack(results_jpeg[results_jpeg[:, 0] == 512, 5])
    tprs_jpeg_256 = np.stack(results_jpeg[results_jpeg[:, 0] == 256, 5])
    tprs_jpeg_128 = np.stack(results_jpeg[results_jpeg[:, 0] == 128, 5])
    tprs_jpeg_64 = np.stack(results_jpeg[results_jpeg[:, 0] == 64, 5])

    fprs_png = np.stack(results_png[:, 4])
    fprs_png_512 = np.stack(results_png[results_png[:, 0] == 512, 4])
    fprs_png_256 = np.stack(results_png[results_png[:, 0] == 256, 4])
    fprs_png_128 = np.stack(results_png[results_png[:, 0] == 128, 4])
    fprs_png_64 = np.stack(results_png[results_png[:, 0] == 64, 4])

    tprs_png = np.stack(results_png[:, 5])
    tprs_png_512 = np.stack(results_png[results_png[:, 0] == 512, 5])
    tprs_png_256 = np.stack(results_png[results_png[:, 0] == 256, 5])
    tprs_png_128 = np.stack(results_png[results_png[:, 0] == 128, 5])
    tprs_png_64 = np.stack(results_png[results_png[:, 0] == 64, 5])

    # Save results
    if not os.path.exists('results/'):
        os.makedirs('results/')

    np.save('results/' + results_jpeg_path, results_jpeg)
    np.save('results/' + results_png_path, results_png)

# Load results
else:
    results_jpeg = np.load('results/' + results_jpeg_path, allow_pickle=True)
    results_png = np.load('results/' + results_png_path, allow_pickle=True)

    fprs_jpeg = np.stack(results_jpeg[:, 4])
    fprs_jpeg_512 = np.stack(results_jpeg[(results_jpeg[:, 0] == 512) & (results_jpeg[:, 2] >= 90), 4])
    fprs_jpeg_256 = np.stack(results_jpeg[results_jpeg[:, 0] == 256, 4])
    fprs_jpeg_128 = np.stack(results_jpeg[results_jpeg[:, 0] == 128, 4])
    fprs_jpeg_64 = np.stack(results_jpeg[results_jpeg[:, 0] == 64, 4])

    tprs_jpeg = np.stack(results_jpeg[:, 5])
    tprs_jpeg_512 = np.stack(results_jpeg[(results_jpeg[:, 0] == 512) & (results_jpeg[:, 2] >= 90), 5])
    tprs_jpeg_256 = np.stack(results_jpeg[results_jpeg[:, 0] == 256, 5])
    tprs_jpeg_128 = np.stack(results_jpeg[results_jpeg[:, 0] == 128, 5])
    tprs_jpeg_64 = np.stack(results_jpeg[results_jpeg[:, 0] == 64, 5])

    fprs_png = np.stack(results_png[:, 4])
    fprs_png_512 = np.stack(results_png[results_png[:, 0] == 512, 4])
    fprs_png_256 = np.stack(results_png[results_png[:, 0] == 256, 4])
    fprs_png_128 = np.stack(results_png[results_png[:, 0] == 128, 4])
    fprs_png_64 = np.stack(results_png[results_png[:, 0] == 64, 4])

    tprs_png = np.stack(results_png[:, 5])
    tprs_png_512 = np.stack(results_png[results_png[:, 0] == 512, 5])
    tprs_png_256 = np.stack(results_png[results_png[:, 0] == 256, 5])
    tprs_png_128 = np.stack(results_png[results_png[:, 0] == 128, 5])
    tprs_png_64 = np.stack(results_png[results_png[:, 0] == 64, 5])

# Mean AUC score
auc_512 = np.mean([np.mean([results_jpeg[(results_jpeg[:, 0] == 512) & (results_jpeg[:, 2] >= 90), 3], np.mean(results_png[results_png[:, 0] == 512, 3])])])
auc_256 = np.mean([np.mean([results_jpeg[(results_jpeg[:, 0] == 256) & (results_jpeg[:, 2] >= 90), 3], np.mean(results_png[results_png[:, 0] == 256, 3])])])
auc_128 = np.mean([np.mean([results_jpeg[(results_jpeg[:, 0] == 128) & (results_jpeg[:, 2] >= 90), 3], np.mean(results_png[results_png[:, 0] == 128, 3])])])
auc_64 = np.mean([np.mean([results_jpeg[(results_jpeg[:, 0] == 64) & (results_jpeg[:, 2] >= 90), 3], np.mean(results_png[results_png[:, 0] == 64, 3])])])


dis = np.nanmean(np.where(results_png[results_png[:, 1] == 4, 3] != 0, results_png[results_png[:, 1] == 4, 3], np.nan))  # TODO
#print(results_png[:, 0])
#print(results_png[:, 1])
#print()
#print(results_png[:, 3])

# FPR, TPR for all images
fprs = np.vstack([fprs_jpeg, fprs_png])
fprs_512 = np.vstack([fprs_jpeg_512, fprs_png_512])
fprs_256 = np.vstack([fprs_jpeg_256, fprs_png_256])
fprs_128 = np.vstack([fprs_jpeg_128, fprs_png_128])
fprs_64 = np.vstack([fprs_jpeg_64, fprs_png_64])

tprs = np.vstack([tprs_jpeg, tprs_png])
tprs_512 = np.vstack([tprs_jpeg_512, tprs_png_512])
tprs_256 = np.vstack([tprs_jpeg_256, tprs_png_256])
tprs_128 = np.vstack([tprs_jpeg_128, tprs_png_128])
tprs_64 = np.vstack([tprs_jpeg_64, tprs_png_64])

fpr, tpr = get_mean_roc(fprs, tprs)
fpr_512, tpr_512 = get_mean_roc(fprs_512, tprs_512)
fpr_256, tpr_256 = get_mean_roc(fprs_256, tprs_256)
fpr_128, tpr_128 = get_mean_roc(fprs_128, tprs_128)
fpr_64, tpr_64 = get_mean_roc(fprs_64, tprs_64)

plot_roc([fpr_512, fpr_256, fpr_128, fpr_64], [tpr_512, tpr_256, tpr_128, tpr_64], np.asarray([auc_512, auc_256, auc_128, auc_64]), show=True, save=False)


# TODO fai ROC solo per png
