import numpy as np
import os
import random
import sys
import tqdm
from argparse import Namespace
from main import main

# Setup
args = Namespace()
args.interpolate = False
args.show = False
args.save = False
args.show_diff_plot = False
args.save_diff_plot = False
dir_path = 'img/'  # Directory containing images
n = 3  # Number of images to be tested

# Array of values to be tried
win_sizes = [64, 128, 256]
stop_thresholds = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
probs_r_b_in_c1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# File list
valid_extensions = ['.jpeg', '.jpg', '.jpe', '.jfif', '.jif', '.png']
file_list = [dir_path + img_file for img_file in os.listdir(dir_path) for ext in valid_extensions if img_file.endswith(ext)]
imgs = random.sample(file_list, n)

# Results list
results = []

# Test algorithm
progress_bar = tqdm.tqdm(total=len(win_sizes)*len(stop_thresholds)*len(probs_r_b_in_c1)*n)
sys.stdout = open(os.devnull, 'w')  # Suppress calls to print()


for i in imgs:
    for ws in win_sizes:
        for st in stop_thresholds:
            for p in probs_r_b_in_c1:
                args.img_path = i
                args.win_size = ws
                args.stop_threshold = st
                args.prob_r_b_in_c1 = p
                _, _, auc = main(args)

                # Append results
                results.append([ws, st, p, auc])

                # Update progress bar
                progress_bar.update(1)

sys.stdout = sys.__stdout__  # Enable calls to print()

# Save results
save = True
if save:
    np.save('variables_test_results_raw.npy', np.asarray(results))

# Load existing data
results_path = ''
if results_path != '':
    results = np.load(results_path)

# TODO Optimize results matrix to choose best parameters combination
