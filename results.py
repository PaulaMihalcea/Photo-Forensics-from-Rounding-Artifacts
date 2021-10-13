import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import tqdm
from argparse import ArgumentParser, Namespace
from datetime import datetime
from main import main as mm
from utils import get_filename, get_file_list, get_last_directory


# Get image information from filename
def get_image_info(filename, extension):
    # Split filename
    el = filename.split('_')
    original_filename = ''

    # Extract information
    if extension in ['png', 'PNG']:
        for e in el[0:-2]:
            original_filename = original_filename + '_' + e
        original_filename = original_filename[1:]
        quality = -1
        manip_size = int(el[-1])
        manip_type = el[-2]
    elif extension in ['jpeg', 'jpg', 'jpe', 'jfif', 'jif', 'JPEG', 'JPG', 'JPE', 'JFIF', 'JIF']:
        for e in el[0:-3]:
            original_filename = original_filename + '_' + e
        original_filename = original_filename[1:]
        quality = int(el[-1])
        manip_size = int(el[-2])
        manip_type = el[-3]
    else:
        raise ValueError('Invalid file extension (allowed extensions: .png, .jpeg, .jpg, .jpe, .jfif or .jif.)')

    return original_filename, manip_size, manip_type, quality


# Get dimples strength from report file
def get_dimples_strength(dimples_df, filename):
    if dimples_df is not None:
        dimples_strength = dimples_df[dimples_df.img_name == filename].dimples_strength.values
        if len(dimples_strength) == 0:
            dimples_strength = None
        else:
            dimples_strength = dimples_strength[0]
    else:
        dimples_strength = None

    return dimples_strength


# Manipulation size data partition for AUC plots
# Returns the average AUC for each manipulation size of the given dataframe
def get_manip_size_partition_auc_mean(dataframe):
    # Initialization
    manip_sizes = [512, 128, 256, 512]
    partitions = []

    # Generate dataframes
    for size in manip_sizes:
        partitions.append(dataframe[dataframe['manip_size'] == size]['auc'].mean(skipna=True))

    return partitions


# Plot average ROC curve
def plot_avg_roc(results_fpr, results_tpr, roc_64, roc_128, roc_256, roc_512):

    # Group FPR and TPR by image, then transform into NumPy arrays
    fpr = results_fpr.groupby('img_name')['fpr'].apply(np.array)
    tpr = results_tpr.groupby('img_name')['tpr'].apply(np.array)

    # Get mean curves
    roc_64_fpr_mean = np.sort(fpr[roc_64['img_name']].mean())
    roc_64_tpr_mean = np.sort(tpr[roc_64['img_name']].mean())
    roc_128_fpr_mean = np.sort(fpr[roc_128['img_name']].mean())
    roc_128_tpr_mean = np.sort(tpr[roc_128['img_name']].mean())
    roc_256_fpr_mean = np.sort(fpr[roc_256['img_name']].mean())
    roc_256_tpr_mean = np.sort(tpr[roc_256['img_name']].mean())
    roc_512_fpr_mean = np.sort(fpr[roc_512['img_name']].mean())
    roc_512_tpr_mean = np.sort(tpr[roc_512['img_name']].mean())

    # Average AUC
    avg_auc = [auc(roc_512_fpr_mean, roc_512_tpr_mean), auc(roc_256_fpr_mean, roc_256_tpr_mean), auc(roc_128_fpr_mean, roc_128_tpr_mean), auc(roc_64_fpr_mean, roc_64_tpr_mean)]

    # Base plot
    plt.plot(roc_512_fpr_mean * 100, roc_512_tpr_mean * 100)
    plt.plot(roc_256_fpr_mean * 100, roc_256_tpr_mean * 100)
    plt.plot(roc_128_fpr_mean * 100, roc_128_tpr_mean * 100)
    plt.plot(roc_64_fpr_mean * 100, roc_64_tpr_mean * 100)

    # Plot display options
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('square')
    plt.grid()

    # Axes labels
    avg_auc_label = str(avg_auc[0])[:4] + ' (512) , ' + str(avg_auc[1])[:4] + ' (256), ' + str(avg_auc[2])[:4] + ' (128), ' + str(avg_auc[3])[:4] + ' (64)'
    plt.xlabel('False Positive (%)'
               '\n\n'
               'AUC score: {}'.format(avg_auc_label), fontname='Chapman-Regular', fontsize=13, labelpad=15)
    plt.ylabel('True Positive (%)', fontname='Chapman-Regular', fontsize=13)
    plt.tight_layout()

    # Ticks
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    plt.tick_params(direction='in', top=True, right=True)

    # Lines
    plt.gca().lines[0].set_color('red')
    plt.gca().lines[0].set_linestyle('dotted')
    plt.gca().lines[0].set_linewidth(1)
    plt.gca().lines[1].set_color('green')
    plt.gca().lines[1].set_linestyle('dashed')
    plt.gca().lines[1].set_linewidth(1)
    plt.gca().lines[2].set_color('blue')
    plt.gca().lines[2].set_linestyle('dashdot')
    plt.gca().lines[2].set_linewidth(1)
    plt.gca().lines[3].set_color('black')
    plt.gca().lines[3].set_linestyle('solid')
    plt.gca().lines[3].set_linewidth(1)

    # Legend
    plt.legend(edgecolor='black', fancybox=False, loc='lower right', prop=fm.FontProperties(family='serif'),
               handlelength=1.5, handletextpad=0.1,
               handles=[mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1, label='512'),
                        mlines.Line2D([], [], color='green', linestyle='dashed', linewidth=1, label='256'),
                        mlines.Line2D([], [], color='blue', linestyle=(0, (3, 1, 1, 1)), linewidth=1, label='128'),
                        mlines.Line2D([], [], color='black', linestyle='solid', linewidth=1, label='64')])

    return plt


# Main results plot function
def plot_results(results, results_fpr, results_tpr, results_filename, dimples_strength=False, show_plots=True, save_plots=True):
    # Main PNG & JPEG dataframes
    png = results[results['format'] == 'png']
    jpeg = results[(results['format'] == 'jpeg')]

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # Fig. 6: Average ROC (PNG + JPEG 91+, partitioned by manipulation size)
    roc_64 = png[png['manip_size'] == 64].append(jpeg[(jpeg['manip_size'] == 64) & (jpeg['quality'] >= 91)])
    roc_128 = png[png['manip_size'] == 128].append(jpeg[(jpeg['manip_size'] == 128) & (jpeg['quality'] >= 91)])
    roc_256 = png[png['manip_size'] == 256].append(jpeg[(jpeg['manip_size'] == 256) & (jpeg['quality'] >= 91)])
    roc_512 = png[png['manip_size'] == 512].append(jpeg[(jpeg['manip_size'] == 512) & (jpeg['quality'] >= 91)])

    # Plot
    plt.figure('Average ROC', figsize=(10, 6))
    plot_avg_roc(results_fpr, results_tpr, roc_64, roc_128, roc_256, roc_512)

    # Save/show plot
    if save_plots:
        plt.savefig(results_filename + '_roc_plot.png', dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # Fig. 7(a): AUC by manipulation type, partitioned by manipulation size
    copy_move = png[png['manip_type'] == 'copy-move']
    median_filter = png[png['manip_type'] == 'median-filter']
    rotate = png[png['manip_type'] == 'rotation']
    content_aware_fill = png[png['manip_type'] == 'content-aware-fill']

    # Manipulation type average AUC
    copy_move_means = get_manip_size_partition_auc_mean(copy_move)
    median_filter_means = get_manip_size_partition_auc_mean(median_filter)
    rotate_means = get_manip_size_partition_auc_mean(rotate)
    content_aware_fill_means = get_manip_size_partition_auc_mean(content_aware_fill)

    # Plot
    width = 0.19
    diff = 0.05
    plt.figure('Manipulation type AUC', figsize=(10, 6))
    manip_type_index = np.arange(4)
    plt.bar(manip_type_index, [copy_move_means[0], median_filter_means[0], rotate_means[0], content_aware_fill_means[0]], width - diff, color='#ffffff', edgecolor='black')
    plt.bar(manip_type_index + width, [copy_move_means[1], median_filter_means[1], rotate_means[1], content_aware_fill_means[1]], width - diff, color='#c2c2c2', edgecolor='black')
    plt.bar(manip_type_index + width * 2, [copy_move_means[2], median_filter_means[2], rotate_means[2], content_aware_fill_means[2]], width - diff, color='#7d7d7d', edgecolor='black')
    plt.bar(manip_type_index + width * 3, [copy_move_means[3], median_filter_means[3], rotate_means[3], content_aware_fill_means[3]], width - diff, color='#363636', edgecolor='black')

    plt.ylabel('AUC', fontname='Chapman-Regular', fontsize=16)
    plt.xticks(manip_type_index + (width + diff * 4) * 4 / 4, ('copy-move', 'median filter', 'rotate', 'region fill'), fontname='Chapman-Regular', fontsize=16)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontname='Chapman-Regular', fontsize=16)
    plt.ylim((0.2, 1))

    ax = plt.gca()
    plt.tick_params(direction='in', top=True, right=True)
    ax.set_axisbelow(True)
    ax.grid(color='#e6e6e6')

    # Save/show plot
    if save_plots:
        plt.savefig(results_filename + '_manip_type_plot.png', dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # Fig. 7(b): AUC by EM window size, partitioned by manipulation size
    win_size_256 = png[png['win_size'] == 256]
    win_size_128 = png[png['win_size'] == 128]
    win_size_64 = png[png['win_size'] == 64]

    # EM window size average AUC
    win_size_256_means = get_manip_size_partition_auc_mean(win_size_256)
    win_size_128_means = get_manip_size_partition_auc_mean(win_size_128)
    win_size_64_means = get_manip_size_partition_auc_mean(win_size_64)

    # Plot
    width = 0.19
    diff = 0.05
    plt.figure('Window size AUC', figsize=(10, 6))
    win_size_index = np.arange(3)
    plt.bar(win_size_index, [win_size_256_means[0], win_size_128_means[0], win_size_64_means[0]], width - diff, color='#ffffff', edgecolor='black')
    plt.bar(win_size_index + width, [win_size_256_means[1], win_size_128_means[1], win_size_64_means[1]], width - diff, color='#c2c2c2', edgecolor='black')
    plt.bar(win_size_index + width * 2, [win_size_256_means[2], win_size_128_means[2], win_size_64_means[2]], width - diff, color='#7d7d7d', edgecolor='black')
    plt.bar(win_size_index + width * 3, [win_size_256_means[3], win_size_128_means[3], win_size_64_means[3]], width - diff, color='#363636', edgecolor='black')

    plt.xlabel('Window size', fontname='Chapman-Regular', fontsize=16, labelpad=5)
    plt.ylabel('AUC', fontname='Chapman-Regular', fontsize=13)
    plt.xticks(win_size_index + (width + diff * 4) * 4 / 4, ('256', '128', '64'), fontname='Chapman-Regular', fontsize=16)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontname='Chapman-Regular', fontsize=16)
    plt.ylim((0.2, 1))

    ax = plt.gca()
    plt.tick_params(direction='in', top=True, right=True)
    ax.set_axisbelow(True)
    ax.grid(color='#e6e6e6')

    # Save/show plot
    if save_plots:
        plt.savefig(results_filename + '_win_size_plot.png', dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # Fig. 7(c): AUC by JPEG quality, partitioned by manipulation size
    jpeg_60 = jpeg[(jpeg['quality'] >= 60) & (jpeg['quality'] <= 70)]
    jpeg_71 = jpeg[(jpeg['quality'] >= 71) & (jpeg['quality'] <= 80)]
    jpeg_81 = jpeg[(jpeg['quality'] >= 81) & (jpeg['quality'] <= 90)]
    jpeg_91 = jpeg[jpeg['quality'] >= 91]

    # JPEG quality average AUC
    jpeg_60_means = get_manip_size_partition_auc_mean(jpeg_60)
    jpeg_71_means = get_manip_size_partition_auc_mean(jpeg_71)
    jpeg_81_means = get_manip_size_partition_auc_mean(jpeg_81)
    jpeg_91_means = get_manip_size_partition_auc_mean(jpeg_91)

    # Plot
    width = 0.19
    diff = 0.05
    plt.figure('JPEG quality AUC', figsize=(10, 6))
    jpeg_index = np.arange(4)
    plt.bar(jpeg_index, [jpeg_60_means[0], jpeg_71_means[0], jpeg_81_means[0], jpeg_91_means[0]], width - diff, color='#ffffff', edgecolor='black')
    plt.bar(jpeg_index + width, [jpeg_60_means[1], jpeg_71_means[1], jpeg_81_means[1], jpeg_91_means[1]], width - diff, color='#c2c2c2', edgecolor='black')
    plt.bar(jpeg_index + width*2, [jpeg_60_means[2], jpeg_71_means[2], jpeg_81_means[2], jpeg_91_means[2]], width - diff, color='#7d7d7d', edgecolor='black')
    plt.bar(jpeg_index + width*3, [jpeg_60_means[3], jpeg_71_means[3], jpeg_81_means[3], jpeg_91_means[3]], width - diff, color='#363636', edgecolor='black')

    plt.xlabel('JPEG Quality', fontname='Chapman-Regular', fontsize=14, labelpad=5)
    plt.ylabel('AUC', fontname='Chapman-Regular', fontsize=16)
    plt.xticks(jpeg_index + (width + diff * 4) * 4 / 4, ('60-70', '71-80', '81-90', '91-100'), fontname='Chapman-Regular', fontsize=16)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontname='Chapman-Regular', fontsize=16)
    plt.ylim((0.2, 1))

    ax = plt.gca()
    plt.tick_params(direction='in', top=True, right=True)
    ax.set_axisbelow(True)
    ax.grid(color='#e6e6e6')

    # Save/show plot
    if save_plots:
        plt.savefig(results_filename + '_jpeg_quality_plot.png', dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

    # Average ROCs by dimple strength; only available if dimple data is known
    if dimples_strength:

        # Average ROC by dimples strength: low intensity (15-30)
        roc_64_lo = roc_64[(roc_64['dimples_strength'] >= 15) & (roc_64['dimples_strength'] < 30)]
        roc_128_lo = roc_128[(roc_128['dimples_strength'] >= 15) & (roc_128['dimples_strength'] < 30)]
        roc_256_lo = roc_256[(roc_256['dimples_strength'] >= 15) & (roc_256['dimples_strength'] < 30)]
        roc_512_lo = roc_512[(roc_512['dimples_strength'] >= 15) & (roc_512['dimples_strength'] < 30)]

        # Plot
        plt.figure('Average ROC by dimples: low strength', figsize=(10, 6))
        plot_avg_roc(results_fpr, results_tpr, roc_64_lo, roc_128_lo, roc_256_lo, roc_512_lo)

        # Save/show plot
        if save_plots:
            plt.savefig(results_filename + '_roc_dimples_lo_plot.png', dpi=300)
        if show_plots:
            plt.show()
        plt.close()

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

        # Average ROC by dimples strength: medium intensity (30-45)
        roc_64_md = roc_64[(roc_64['dimples_strength'] >= 30) & (roc_64['dimples_strength'] < 45)]
        roc_128_md = roc_128[(roc_128['dimples_strength'] >= 30) & (roc_128['dimples_strength'] < 45)]
        roc_256_md = roc_256[(roc_256['dimples_strength'] >= 30) & (roc_256['dimples_strength'] < 45)]
        roc_512_md = roc_512[(roc_512['dimples_strength'] >= 30) & (roc_512['dimples_strength'] < 45)]

        # Plot
        plt.figure('Average ROC by dimples: medium strength', figsize=(10, 6))
        plot_avg_roc(results_fpr, results_tpr, roc_64_md, roc_128_md, roc_256_md, roc_512_md)

        # Save/show plot
        if save_plots:
            plt.savefig(results_filename + '_roc_dimples_md_plot.png', dpi=300)
        if show_plots:
            plt.show()
        plt.close()

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------

        # Average ROC by dimples strength: high intensity (45+)
        roc_64_hi = roc_64[roc_64['dimples_strength'] >= 45]
        roc_128_hi = roc_128[roc_128['dimples_strength'] >= 45]
        roc_256_hi = roc_256[roc_256['dimples_strength'] >= 45]
        roc_512_hi = roc_512[roc_512['dimples_strength'] >= 45]

        # Plot
        plt.figure('Average ROC by dimples: high strength', figsize=(10, 6))
        plot_avg_roc(results_fpr, results_tpr, roc_64_hi, roc_128_hi, roc_256_hi, roc_512_hi)

        # Save/show plot
        if save_plots:
            plt.savefig(results_filename + '_roc_dimples_hi_plot.png', dpi=300)
        if show_plots:
            plt.show()
        plt.close()


def main(args):
    # Welcome message
    print('Photo Forensics from Rounding Artifacts: results script')
    print('Author: Paula Mihalcea')
    print('Version: 1.0')
    print('Based on a research by S. Agarwal and H. Farid. Details & source code at https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts.')

    # Dimples strength dataframe loading
    if os.path.isfile('results/report.csv'):
        dimples_df = pd.read_csv('results/report.csv')
    else:
        dimples_df = None

    # Analyze images and save results
    if args.generate:

        # Get file list & shuffle
        file_list = get_file_list(args.dir_path)
        random.shuffle(file_list)

        # Create results subfolder
        if not os.path.exists('results/'):
            os.makedirs('results/')

        # Dataframe creation
        results = pd.DataFrame(columns=['img_name', 'dimples_strength', 'format', 'quality', 'manip_type', 'manip_size', 'win_size', 'auc'])
        results_fpr = pd.DataFrame(columns=['img_name', 'fpr'])
        results_tpr = pd.DataFrame(columns=['img_name', 'tpr'])

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

        results_path = 'results/results_' + get_last_directory(args.dir_path) + '_' + str(args.win_size) + '_' + timestamp + '.csv'
        results_path_fpr = 'results/results_' + get_last_directory(args.dir_path) + '_' + str(args.win_size) + '_' + timestamp + '_fpr.csv'
        results_path_tpr = 'results/results_' + get_last_directory(args.dir_path) + '_' + str(args.win_size) + '_' + timestamp + '_tpr.csv'

        results.to_csv(results_path, index=False)
        results_fpr.to_csv(results_path_fpr, index=False)
        results_tpr.to_csv(results_path_tpr, index=False)

        '''
        img_name: Image name (without extension).
        dimples: JPEG dimples strength (previously calculated; None otherwise).
        format: Image format (0: PNG, 1: JPEG).
        quality: JPEG quality; -1: PNG image.
        manip_type: Manipulation type.
        manip_size: Manipulation ROI size.
        win_size: EM algorithm window size.
        auc: AUC score.
        fpr: False positive rate (FPR); saved in a separate file.
        tpr: True positive rate (TPR); saved in a separate file.
        '''

        # Main setup (uses default parameters)
        args_mm = Namespace()
        args_mm.win_size = args.win_size
        args_mm.stop_threshold = 1e-3
        args_mm.prob_r_b_in_c1 = 0.5
        args_mm.interpolate = False
        args_mm.show = False
        args_mm.save = False
        args_mm.show_roc_plot = False
        args_mm.save_roc_plot = False
        args_mm.show_diff_plot = False
        args_mm.save_diff_plot = False
        args_mm.verbose = False

        imgs_tot = len(file_list)
        imgs_done = 0

        # Progress bar
        progress_bar = tqdm.tqdm(total=imgs_tot)

        # Time
        start = time.time()

        # Main loop
        for i in file_list:
            # Basic image information
            filename, extension = get_filename(i)

            try:  # Too broad exception clause? Definitely. Perfect for avoiding errors in a script that might be up and running for days? Absolutely.
                # More image information
                args_mm.img_path = args.dir_path + '/' + i
                original_filename, manip_size, manip_type, quality = get_image_info(filename, extension)
                dimples_strength = get_dimples_strength(dimples_df, original_filename)

                # Update progress bar
                progress_bar.set_description('Processing image {}'.format(filename + '.{}'.format(extension)))

                # Main EM algorithm
                _, auc, fpr, tpr = mm(args_mm)

                # Save results
                results_dict = {'img_name': filename, 'dimples_strength': dimples_strength, 'format': extension, 'quality': quality, 'manipulation_type': manip_type, 'manip_size': manip_size, 'win_size': args.win_size, 'auc': auc}
                results_fpr_dict = {'img_name': filename, 'fpr': fpr}
                results_tpr_dict = {'img_name': filename, 'tpr': tpr}

                results = pd.DataFrame(results_dict, index=[0])
                results_fpr = pd.DataFrame(results_fpr_dict)
                results_tpr = pd.DataFrame(results_tpr_dict)

                results.to_csv(results_path, mode='a', header=False)
                results_fpr.to_csv(results_path_fpr, mode='a', header=False)
                results_tpr.to_csv(results_path_tpr, mode='a', header=False)

                # Update progress bar
                progress_bar.update(1)
                imgs_done += 1

            except Exception as e:
                print('An exception happened while analyzing file' + filename + '.' + extension + ':')
                print(getattr(e, 'message', repr(e)))

                # Update progress bar nonetheless
                progress_bar.update(1)

                continue

        end = time.time()

        # Final message
        print()
        print('Summary')
        if imgs_done == imgs_tot:
            print('Images analyzed: {}.'.format(imgs_tot))
        else:
            print('Images analyzed: {}/'.format(imgs_done) + '{} (missing images have generated errors).'.format(imgs_tot))
        # Elapsed time
        hours, remaining_time = divmod(end - start, 3600)
        minutes, seconds = divmod(remaining_time, 60)
        if hours >= 1:
            print(
                'Elapsed time: {:.0f}h'.format(hours) + ' {:.0f}m'.format(minutes) + ' {:.2f}s.'.format(seconds))
        elif minutes >= 1:
            print('Elapsed time: {:.0f}m'.format(minutes) + ' {:.2f}s.'.format(seconds))
        else:
            print('Elapsed time: {:.2f}s.'.format(seconds))
        print('Done.')

    else:
        # Time
        start = time.time()

        # Load dataframes
        results = pd.read_csv(args.res_path)
        results_fpr = pd.read_csv(args.res_path.replace('.csv', '_fpr.csv'))
        results_tpr = pd.read_csv(args.res_path.replace('.csv', '_tpr.csv'))

        # Generate plots
        if dimples_df is not None:
            plot_results(results, results_fpr, results_tpr, args.res_path.replace('.csv', ''), True, args.show_plots, args.save_plots)
        else:
            plot_results(results, results_fpr, results_tpr, args.res_path.replace('.csv', ''), False, args.show_plots, args.save_plots)

        end = time.time()

        # Final message
        print()
        print('Summary')
        print('Loaded data from {} images.'.format(len(results)))

        # Elapsed time
        hours, remaining_time = divmod(end - start, 3600)
        minutes, seconds = divmod(remaining_time, 60)
        if hours >= 1:
            print('Elapsed time: {:.0f}h'.format(hours) + ' {:.0f}m'.format(minutes) + ' {:.2f}s.'.format(seconds))
        elif minutes >= 1:
            print('Elapsed time: {:.0f}m'.format(minutes) + ' {:.2f}s.'.format(seconds))
        else:
            print('Elapsed time: {:.2f}s.'.format(seconds))
        print('Done.')

        pass

    return


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Results script for the "Photo Forensics from Rounding Artifacts" project; generates results by analyzing JPEG and PNG images from two given directories (e.g. "path/to/jpeg_images/" and "path/to/png_images/") as described in the referenced paper, or loads existing results from the "results/" folder.')

    # Add parser arguments
    parser.add_argument('generate', help='Analyze images and generate results; only loads existing results if False (default: False).')
    parser.add_argument('-dpath', '--dir_path', help='Path of the directory containing the images to be analyzed. Only needed if generate is True.')
    parser.add_argument('-ws', '--win_size', type=int, help='Window size in pixel (default: 256). Note: must be a multiple of 8. Only needed if generate is True.')
    parser.add_argument('-rpath', '--res_path', help='Path of the CSV results file to be loaded. Only needed if generate is False.')
    parser.add_argument('-shpl', '--show_plots', help='Show the results\' plots (default: True).')
    parser.add_argument('-svpl', '--save_plots', help='Save the results\' plots in the results\' folder (default: True).')

    args = parser.parse_args()

    if args.generate == 'True':
        args.generate = True
    else:
        args.generate = False

    if args.dir_path is not None:
        # Ensure folder path ends with "/"
        if args.dir_path[-1] != '/':
            args.dir_path += '/'
    else:
        args.dir_path = ''

    if args.win_size is None:
        args.win_size = 64

    if args.res_path is not None:
        # Check & adjust filename
        if not args.res_path.endswith('.csv') or not not args.res_path.endswith('.CSV'):
            args.res_path += '.csv'
        if args.res_path.endswith('.CSV'):
            filename = args.res_path.replace('.CSV', '.csv')
    else:
        args.res_path = ''

    if args.show_plots == 'False':
        args.show_plots = False
    else:
        args.show_plots = True

    if args.save_plots == 'False':
        args.save_plots = False
    else:
        args.save_plots = True

    # Run main script
    main(args)
