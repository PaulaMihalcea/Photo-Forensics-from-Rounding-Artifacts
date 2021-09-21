import cv2
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from preprocessing import load_image


# Main ROC & AUC function
def get_roc_auc(img_path, output_map):
    # Load ground truth image
    img_ground_truth = load_image(get_img_ground_truth_path(img_path))

    # Thresholding & normalization
    output_map_norm = np.where(output_map > 0.8, 1, 0).astype(np.uint8)
    output_map_norm = cv2.normalize(output_map_norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Normalize output map before saving

    # ROC curve
    fpr, tpr, _ = roc_curve(img_ground_truth, output_map)

    # AUC score
    auc = roc_auc_score(fpr, tpr)

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
