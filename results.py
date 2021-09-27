import cv2
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from preprocessing import load_image


# Main ROC & AUC function
def get_roc_auc(img_path, output_map):
    # Load ground truth image
    img_ground_truth = load_image(get_img_ground_truth_path(img_path))

    # Thresholding & normalization
    img_ground_truth = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
    img_ground_truth = np.where(img_ground_truth == 255, 1, 0).astype(np.uint8)

    # Flattening
    img_ground_truth = img_ground_truth.flatten()
    output_map = output_map.flatten()

    # ROC curve
    fpr, tpr, _ = roc_curve(img_ground_truth, output_map)

    # AUC score
    auc = roc_auc_score(img_ground_truth, output_map)

    return fpr, tpr, auc


# Ground truth image path generator (from original image path)
def get_img_ground_truth_path(img_path):

    img_name = img_path.split('/')[-1]

    img_ground_truth_name = img_name.split('.')
    img_ground_truth_name = img_ground_truth_name[0] + '_gt.' + img_ground_truth_name[1]

    img_ground_truth_path = ''
    for el in img_path.split('/')[:-1]:
        img_ground_truth_path += '/' + el

    img_ground_truth_path += '/' + img_ground_truth_name
    img_ground_truth_path = img_ground_truth_path[1:]

    return img_ground_truth_path


# ROC curve & AUC score display
def plot_roc(fpr, tpr, auc):
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
    plt.xlabel('False Positive (%)', fontname='Chapman-Regular', fontsize=13)
    plt.ylabel('True Positive (%)', fontname='Chapman-Regular', fontsize=13)

    # Ticks
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')
    plt.tick_params(direction='in', top=True, right=True)

    # Legend
    plt.legend(edgecolor='black', fancybox=False, prop=fm.FontProperties(family='serif'), handlelength=1.5, handletextpad=0.1,
               handles=[mlines.Line2D([], [], color='red', linestyle='dotted', linewidth=1, label='512'),
                        mlines.Line2D([], [], color='green', linestyle='dashed', linewidth=1, label='256'),
                        mlines.Line2D([], [], color='blue', linestyle=(0, (3, 1, 1, 1)), linewidth=1, label='128'),
                        mlines.Line2D([], [], color='black', linestyle='solid', linewidth=1, label='64')])

    # Line 512
    plt.gca().lines[0].set_color('red')
    plt.gca().lines[0].set_linestyle('dotted')
    plt.gca().lines[0].set_linewidth(1)

    # TODO
    '''
    # Line 256
    plt.gca().lines[1].set_color('green')
    plt.gca().lines[1].set_linestyle('dashed')
    plt.gca().lines[0].set_linewidth(1)

    # Line 128
    plt.gca().lines[2].set_color('blue')
    plt.gca().lines[2].set_linestyle('dashdot')
    plt.gca().lines[0].set_linewidth(1)

    # Line 64
    plt.gca().lines[3].set_color('black')
    plt.gca().lines[3].set_linestyle('solid')
    plt.gca().lines[0].set_linewidth(1)
    '''

    # Show plot
    plt.show()

    return
