import cv2
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from interpolate_missing_pixels import interpolate_missing_pixels
from utils import get_filename, get_img_ground_truth_path, get_subfolder, load_image


# Get output map
def get_output_map(prob_b_in_c1_r, blocks_map, img_w, img_h, show=False, save=False, img_path=None, win_size=None, stop_threshold=None, interpolate=False):

    # Initialize empty map
    output_map = np.empty((img_h, img_w, 2))

    for w in blocks_map:  # For each element in the window list...
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 0] += prob_b_in_c1_r[w[2]]
        output_map[w[1]:w[1] + win_size, w[0]:w[0] + win_size, 1] += 1

    for i in range(0, output_map.shape[0]):  # Average
        for j in range(0, output_map.shape[1]):
            output_map[i, j, 0] = output_map[i, j, 0] / output_map[i, j, 1]

    output_map = 1 - output_map[:, :, 0]  # Because the map computed so far actually shows the probability that a pixel has not been modified

    # Replace NaNs...
    if interpolate:  # ...using interpolation...
        output_mask = np.ma.masked_invalid(output_map).mask
        output_map = interpolate_missing_pixels(output_map, output_mask, 'linear')
    else:  # ...or with a neutral probability (0.5)
        output_map = np.nan_to_num(output_map, nan=0.5)

    '''
    # Matplotlib output map plot (for debug purposes only)
    plt.imshow(output_map)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()
    '''

    # Thresholding & normalization
    output_map_norm = np.where(output_map > 0.8, 255, 0).astype(np.uint8)  # Pixels with probability of being manipulated lower than 80% are masked

    # Show output map and/or save it to disk if requested
    filename, extension = get_filename(img_path)
    if show:
        cv2.namedWindow(filename + '.' + extension + ' output map', cv2.WINDOW_NORMAL)
        cv2.imshow(filename + '.' + extension + ' output map', output_map_norm)
        cv2.waitKey(0)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        cv2.imwrite(res_path + '/' + filename + '.png', output_map_norm)

    return output_map_norm


# Plot difference between successive estimates of template c
def get_template_difference_plot(diff_history, show=False, save=False, img_path=None, win_size=None, stop_threshold=None):

    # Create plot
    plt.plot(diff_history)
    plt.xlabel('EM iteration')
    plt.xticks(range(0, len(diff_history)))
    plt.ylabel('Average of the difference matrix between successive estimates of c')

    # Save plot to disk (if requested, otherwise just show it)
    if save:
        filename, extension = get_filename(img_path)
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        plt.savefig(res_path + '/' + filename + '_c_diff_plot.png')
    if show:
        plt.show()

    return


# Main ROC & AUC function
def get_roc_auc(img_path, output_map):
    # Load ground truth image
    img_ground_truth = load_image(get_img_ground_truth_path(img_path), raise_IO=False)

    # No ground truth image exists
    if img_ground_truth is None:
        return None, None, None

    # Ground truth image exists
    else:
        # Thresholding & normalization
        img_ground_truth = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
        img_ground_truth = np.where(img_ground_truth == 255, 1, 0).astype(np.uint8)

        # Flattening
        img_ground_truth = img_ground_truth.flatten()
        output_map = output_map.flatten()

        print('inf:', np.isinf(img_ground_truth).any(), 'nan:', np.isnan(img_ground_truth).any())  # TODO

        # ROC curve
        fpr, tpr, _ = roc_curve(img_ground_truth, output_map)

        # AUC score
        auc = roc_auc_score(img_ground_truth, output_map)

        return fpr, tpr, auc


# ROC curve & AUC score display
def plot_roc(fpr, tpr, auc, show=False, save=False, img_path='', win_size=None, stop_threshold=None):
    fpr = fpr * 100
    tpr = tpr * 100

    # Base plot
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot()

    # Plot display options
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('square')
    plt.grid()

    # Axes labels
    plt.xlabel('False Positive (%)'
               '\n\n'
               'AUC score: {:.2f}'.format(auc), fontname='Chapman-Regular', fontsize=13, labelpad=15)
    plt.ylabel('True Positive (%)', fontname='Chapman-Regular', fontsize=13)
    plt.tight_layout()

    # Ticks
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    plt.tick_params(direction='in', top=True, right=True)

    # Legend
    if win_size == 512:
        line = mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1, label=str(win_size))
        plt.gca().lines[0].set_color('red')
        plt.gca().lines[0].set_linestyle('dotted')
        plt.gca().lines[0].set_linewidth(1)
    elif win_size == 256:
        line = mlines.Line2D([], [], color='green', linestyle='dashed', linewidth=1, label=str(win_size))
        plt.gca().lines[0].set_color('green')
        plt.gca().lines[0].set_linestyle('dashed')
        plt.gca().lines[0].set_linewidth(1)
    elif win_size == 128:
        line = mlines.Line2D([], [], color='blue', linestyle=(0, (3, 1, 1, 1)), linewidth=1, label=str(win_size))
        plt.gca().lines[0].set_color('blue')
        plt.gca().lines[0].set_linestyle('dashdot')
        plt.gca().lines[0].set_linewidth(1)
    elif win_size == 64:
        line = mlines.Line2D([], [], color='black', linestyle='solid', linewidth=1, label=str(win_size))
        plt.gca().lines[0].set_color('black')
        plt.gca().lines[0].set_linestyle('solid')
        plt.gca().lines[0].set_linewidth(1)
    else:
        line = mlines.Line2D([], [], color='orange', linestyle='solid', linewidth=1, label=str(win_size))
        plt.gca().lines[0].set_color('orange')
        plt.gca().lines[0].set_linestyle('solid')
        plt.gca().lines[0].set_linewidth(1)

    plt.legend(edgecolor='black', fancybox=False, prop=fm.FontProperties(family='serif'), handlelength=1.5, handletextpad=0.1, handles=[line])

    # TODO multiclass legend
    '''
    plt.legend(edgecolor='black', fancybox=False, prop=fm.FontProperties(family='serif'), handlelength=1.5, handletextpad=0.1,
               handles=[mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1, label='512'),
                        mlines.Line2D([], [], color='green', linestyle='dashed', linewidth=1, label='256'),
                        mlines.Line2D([], [], color='blue', linestyle=(0, (3, 1, 1, 1)), linewidth=1, label='128'),
                        mlines.Line2D([], [], color='black', linestyle='solid', linewidth=1, label='64')])
    '''

    # Show plot and/or save it to disk if requested
    filename, extension = get_filename(img_path)
    if show:
        plt.show()
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        plt.savefig(res_path + '/' + filename + '_roc_plot.png')

    return
