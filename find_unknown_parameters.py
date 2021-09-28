import numpy as np
import os
import random
import sys
import tqdm
from argparse import Namespace
from main import main
from utils import get_filename

# Parameters to be tried
n = 10  # Number of images to be tested
stop_thresholds = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
probs_r_b_in_c1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Results file path & loading
results_path = 'unknown_parameters_results.npy'
load = True

if load:  # Load existing results
    results = np.load(results_path)
else:  # Or compute them anew
    # Setup
    args = Namespace()
    args.win_size = 64
    args.interpolate = False
    args.show = False
    args.save = False
    args.show_roc_plot = False
    args.save_roc_plot = False
    args.show_diff_plot = False
    args.save_diff_plot = False

    dir_path = 'img/manip_png/'  # Directory containing images; must end with a slash to avoid errors ("/"). Default is a subfolder named "img".

    # Array of values to be tried

    # File list
    valid_extensions = ['.jpeg', '.jpg', '.jpe', '.jfif', '.jif', '.png']
    file_list = [dir_path + img_file for img_file in os.listdir(dir_path) for ext in valid_extensions if img_file.endswith(ext)]
    imgs = random.sample(file_list, n)

    # Results list
    results = []

    # Test algorithm
    progress_bar = tqdm.tqdm(total=len(stop_thresholds)*len(probs_r_b_in_c1)*n)
    sys.stdout = open(os.devnull, 'w')  # Suppress calls to print()

    for i in imgs:
        progress_bar.set_description('Processing image {}'.format(get_filename(i)[0] + '.{}'.format(get_filename(i)[1])))

        for st in stop_thresholds:
            for p in probs_r_b_in_c1:
                # Parameters
                args.img_path = i
                args.stop_threshold = st
                args.prob_r_b_in_c1 = p

                # EM algorithm
                _, auc = main(args)

                # Append results
                results.append([st, p, auc])

                # Update progress bar
                progress_bar.update(1)

    sys.stdout = sys.__stdout__  # Enable calls to print()

    print()

    # Save results for future use
    save = False
    if save:
        np.save(results_path, np.asarray(results))

# Find best parameters
st_means = []
p_means = []

for st in stop_thresholds:
    st_means.append(np.mean(results[results[:, 0] == st, 2]))

for p in probs_r_b_in_c1:
    p_means.append(np.mean(results[results[:, 1] == p, 2]))

best_st = (np.max(st_means), np.argmax(np.max(st_means)))
best_p = (np.max(p_means), np.argmax(np.max(p_means)))

print('Best stop threshold: {} (AUC score: '.format(results[best_st[1], 0]) + '{}).'.format(best_st[0]))
print('Best probability: {} (AUC score: '.format(results[best_p[1], 1]) + '{}).'.format(best_p[0]))
