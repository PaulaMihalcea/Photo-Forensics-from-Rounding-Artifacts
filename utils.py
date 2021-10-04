import cv2
import os
from decimal import Decimal


# Load image
def load_image(img_path, raise_io=True):
    # Load image
    img = cv2.imread(img_path)

    # Check image correctness
    if img is not None:
        return img
    elif raise_io:
        raise IOError('Error while loading image: invalid image file or image file path.')
    else:
        return None


# Get image size (regardless of number of channels)
def get_image_size(img):
    if len(img.shape) == 3:
        img_h, img_w, _ = img.shape
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        raise RuntimeError('Incorrect input image shape.')

    return img_h, img_w


# Get filename from path
def get_filename(file_path):
    tmp_filename = file_path.split('/')[-1]
    extension = tmp_filename.split('.')[-1]
    tmp_filename = tmp_filename.split('.')[:-1]

    if len(tmp_filename) > 1:
        filename = tmp_filename[0]
        for el in tmp_filename[1:]:
            filename += '.' + el
    else:
        filename = tmp_filename[0]

    return filename, extension


# Ground truth image path generator (from original image path)
def get_img_ground_truth_path(img_path):

    filename, extension = get_filename(img_path)
    if extension in ['jpg', 'jpeg', 'jpe', 'jfif', 'jif', 'JPG', 'JPEG', 'JPE', 'JFIF']:  # Exclude JPEG quality from filename in case of JPEG file with specified quality
        filename = filename.split('_')[:-1]
        img_ground_truth_name = ''
        for el in filename:
            img_ground_truth_name += el + '_'
        img_ground_truth_name = img_ground_truth_name[:-1] + '_gt.png'
    else:
        img_ground_truth_name = filename + '_gt.png'

    img_ground_truth_path = ''
    for el in img_path.split('/')[:-1]:
        img_ground_truth_path += '/' + el

    img_ground_truth_path += '/ground_truth/' + img_ground_truth_name
    img_ground_truth_path = img_ground_truth_path[1:]

    return img_ground_truth_path


# Create results subfolder for current image & configuration & returns its path
def get_subfolder(img_path, win_size, stop_threshold):
    # Create subfolder name
    filename, extension = get_filename(img_path)
    res_path = 'results/' + filename + '_' + extension + '_' + str(win_size) + '_' + '{:.0e}'.format(Decimal(stop_threshold))

    # Create subfolder
    try:
        os.makedirs(res_path)
    except FileExistsError:  # Directory already exists
        pass

    return res_path


# Get JPEG and PNG file list
def get_file_list(dir_path):
    # Ensure folder path ends with "/"
    if dir_path[-1] != '/':
        dir_path += '/'

    # Only get JPEG and PNG images
    valid_extensions = ['jpeg', 'jpg', 'jpe', 'jfif', 'jif', 'JPG', 'JPEG', 'JPE', 'JFIF', 'png', 'PNG']
    file_list = [img_file for img_file in os.listdir(dir_path) for ext in valid_extensions if img_file.endswith(ext)]

    return file_list


# Get last directory in path
def get_last_directory(dir_path):
    if dir_path[-1] != '/':
        last_directory = dir_path.split('/')[-1]
    else:
        last_directory = dir_path.split('/')[-2]

    return last_directory
