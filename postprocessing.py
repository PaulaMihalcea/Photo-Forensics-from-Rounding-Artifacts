import cv2
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
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

    # Thresholding & normalization
    output_map_norm = np.where(output_map > 0.8, 255, 0).astype(np.uint8)  # Pixels with probability of being manipulated lower than 80% are masked
    # output_map_norm = output_map  # Uncomment for quick output map debug

    # Show output map and/or save it to disk if requested
    filename, extension = get_filename(img_path)
    if show:
        cv2.namedWindow(filename + '.' + extension + ' output map', cv2.WINDOW_NORMAL)
        cv2.imshow(filename + '.' + extension + ' output map', output_map_norm)
        cv2.waitKey(0)
    if save:
        res_path = get_subfolder(img_path, win_size, stop_threshold)
        cv2.imwrite(res_path + '/' + filename + '.png', output_map_norm)

    return output_map_norm, output_map


# Plot difference between successive estimates of template c
def get_template_difference_plot(diff_history, show=False, save=False, img_path=None, win_size=None, stop_threshold=None):

    if show or save:
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
def get_roc_auc(img_path, output_map, roc_type):
    # Load ground truth image
    img_ground_truth = load_image(get_img_ground_truth_path(img_path), raise_io=False)

    # No ground truth image exists
    if img_ground_truth is None:
        return None, None, None, None

    # Ground truth image exists
    else:
        # Thresholding & normalization
        img_ground_truth = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
        img_ground_truth = np.where(img_ground_truth == 255, 1, 0).astype(np.uint8)

        # Flattening
        img_ground_truth = img_ground_truth.flatten()
        output_map = output_map.flatten()

        # ROC curve
        if roc_type == 'custom':
            fpr, tpr = roc_curve_custom(img_ground_truth, output_map, np.linspace(0, 1, 50))
            thr = None
        elif roc_type == 'sklearn':
            fpr, tpr, thr = roc_curve(img_ground_truth, output_map)
        else:
            raise ValueError('Invalid ROC function name. ROC curve function can only be "custom" or "sklearn".')

        # AUC score
        try:
            if roc_type == 'custom':
                auc_score = auc(fpr, tpr)
            elif roc_type == 'sklearn':
                auc_score = roc_auc_score(img_ground_truth, output_map)
            else:
                raise ValueError('Invalid ROC function name. ROC curve function can only be "custom" or "sklearn".')
        except:
            auc_score = 0

        return auc_score, fpr, tpr, thr


# ROC curve & AUC score display
def plot_roc(fpr, tpr, auc, show=False, save=False, img_path='', win_size=None, stop_threshold=None):
    # Base plot
    plt.plot(fpr * 100, tpr * 100)

    # Plot display options
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('square')
    plt.grid()

    # Axes labels
    plt.xlabel('False Positive (%)'
               '\n\n'
               'AUC score: {}'.format(auc), fontname='Chapman-Regular', fontsize=13, labelpad=15)
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

        plt.legend(edgecolor='black', fancybox=False, loc='lower right', prop=fm.FontProperties(family='serif'), handlelength=1.5, handletextpad=0.1, handles=[line])

    # Show plot and/or save it to disk if requested
    if img_path != '':
        filename, _ = get_filename(img_path)
    else:
        filename = 'results'

    if save:
        if img_path != '':
            res_path = get_subfolder(img_path, win_size, stop_threshold)
        else:
            res_path = 'results'
        plt.savefig(res_path + '/' + filename + '_roc_plot.png')
    if show:
        plt.show()

    return


# ROC curve
# Original function by StackOverflow user Flavia Giammarino:
# https://stackoverflow.com/a/61323665
# Optimized by Paula Mihalcea
def roc_curve_custom(output_map, ground_truth, thresholds):
    # Initialize FPR & TPR arrays
    fpr = np.empty_like(thresholds)
    tpr = np.empty_like(thresholds)

    # Compute FPR & TPR
    for t in range(0, len(thresholds)):
        y_pred = np.where(ground_truth >= thresholds[t], 1, 0)
        fp = np.sum((y_pred == 1) & (output_map == 0))
        tp = np.sum((y_pred == 1) & (output_map == 1))
        fn = np.sum((y_pred == 0) & (output_map == 1))
        tn = np.sum((y_pred == 0) & (output_map == 0))
        fpr[t] = fp / (fp + tn)
        tpr[t] = tp / (tp + fn)

    return fpr, tpr


# Interpolate missing pixels (NaNs)
# Function by StackOverflow user Sam De Meyer, based on user G M's answer:
# https://stackoverflow.com/a/68558547
def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image
