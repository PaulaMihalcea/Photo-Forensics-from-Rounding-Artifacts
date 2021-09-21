import numpy as np
import os
import sys
from argparse import Namespace
from main import main

# Setup
args = Namespace()
args.interpolate = False
args.show = False
args.save = False
args.show_diff_plot = False
args.save_diff_plot = False

# Array of values to be tried
imgs = ['', '', '']
win_sizes = [64, 128, 256]
stop_thresholds = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
probs_r_b_in_c1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Test algorithm
sys.stdout = open(os.devnull, 'w')  # Suppress calls to print()


for i in imgs:
    for ws in win_sizes:
        for t in stop_thresholds:
            for p in probs_r_b_in_c1:
                args.img_path = imgs[i]
                args.win_size = win_sizes[ws]
                args.stop_threshold = stop_thresholds[t]
                args.prob_r_b_in_c1 = probs_r_b_in_c1[p]
                main(args)
sys.stdout = sys.__stdout__  # Enable calls to print()

# Display results
# TODO
