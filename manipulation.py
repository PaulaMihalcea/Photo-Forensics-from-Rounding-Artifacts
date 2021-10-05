import cv2
import numpy as np
import os
import random
import time
import tqdm
from argparse import ArgumentParser
from PIL import Image
from utils import get_filename, get_file_list, get_image_size, load_image


# Define random manipulation ROI
def manipulate_image(img_path, manipulation, roi_size):
    # Load image
    img = load_image(img_path)
    img_width, img_height = get_image_size(img)

    # Randomly choose manipulation ROI position
    x = random.randint(0, img_height - roi_size)
    y = random.randint(0, img_width - roi_size)

    roi = (x, y, roi_size)

    # Apply manipulation
    # Copy-move
    if manipulation == 'copy-move':
        manip_img, ground_truth = copy_move(img, roi)

    # Median filter
    elif manipulation == 'median-filter':
        manip_img, ground_truth = median_filter(img, roi)

    # Rotation
    elif manipulation == 'rotation':
        manip_img, ground_truth = rotate(img_path, roi)  # Ground truth is replaced as rotation has a round mask which must be calculated separately

    # Content-aware fill
    elif manipulation == 'content-aware-fill':
        manip_img, ground_truth = content_aware_fill(img, x, y, roi_size)

    else:
        raise ValueError('Invalid manipulation method. Possible values: "copy-move", "median-filter", "rotation", "content-aware-fill".')

    return manip_img, ground_truth


# Ground truth
def get_ground_truth(img, x, y, roi_size):
    # Ground truth image
    ground_truth = img.copy().astype(float)
    ground_truth[y:y + roi_size, x:x + roi_size] = np.inf
    ground_truth = np.where(ground_truth == np.inf, 255, 0).astype(np.uint8)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    return ground_truth


# Copy-move
def copy_move(img, roi):
    # Image parameters
    img_width, img_height = get_image_size(img)

    # ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Randomly choose ROI to be copied
    x_copy = random.randint(0, img_height - roi_size)
    y_copy = random.randint(0, img_width - roi_size)

    # Generate manipulated image
    manip_img = img.copy()
    manip_img[y:y + roi_size, x:x + roi_size] = img[y_copy:y_copy + roi_size, x_copy:x_copy + roi_size]

    # Generate ground truth image
    ground_truth = get_ground_truth(img, x, y, roi_size)

    return manip_img, ground_truth


# Median filter
def median_filter(img, roi):
    # ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Manipulate ROI
    manip_roi = cv2.medianBlur(img[y:y + roi_size, x:x + roi_size], 3)

    # Generate manipulated image
    manip_img = img.copy()
    manip_img[y:y+roi_size, x:x+roi_size] = manip_roi

    # Generate ground truth image
    ground_truth = get_ground_truth(img, x, y, roi_size)

    return manip_img, ground_truth


# Rotation
def rotate(img_path, roi):
    # Load image (PIL)
    img = Image.open(img_path).convert('RGBA')
    width, height = img.size

    # Create ground truth image
    ground_truth = Image.fromarray(np.zeros((height, width, 4)), 'RGBA')

    # Define ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Randomly choose rotation degrees
    degrees = random.randint(10, 80)

    # Create rotation mask
    cropped_roi = np.asarray(img.crop(box=(x - roi_size, y - roi_size, x + roi_size + 1, y + roi_size + 1)))
    rot_mask = np.zeros((2 * roi_size + 1, 2 * roi_size + 1))

    for i in range(cropped_roi.shape[0]):
        for j in range(cropped_roi.shape[1]):
            if (i - roi_size) ** 2 + (j - roi_size) ** 2 <= roi_size ** 2:
                rot_mask[i, j] = 1

    # Create manipulated region
    manip_roi = np.empty(cropped_roi.shape, dtype='uint8')
    manip_roi[:, :, :3] = cropped_roi[:, :, :3]
    manip_roi[:, :, 3] = rot_mask * 255
    manip_roi = Image.fromarray(manip_roi, 'RGBA').rotate(degrees)

    # Create ground truth manipulated region
    cropped_roi = np.ones(cropped_roi.shape) * 255
    gt_manip_roi = np.empty(cropped_roi.shape, dtype='uint8')
    gt_manip_roi[:, :, :3] = cropped_roi[:, :, :3]
    gt_manip_roi[:, :, 3] = rot_mask * 255
    gt_manip_roi = Image.fromarray(gt_manip_roi, 'RGBA').rotate(degrees)

    # Generate manipulated image
    manip_img = img.copy()
    manip_img.paste(manip_roi, (x - roi_size, y - roi_size, x + roi_size + 1, y + roi_size + 1)[:2], manip_roi.convert('RGBA'))

    # Generate manipulated ground truth image
    ground_truth.paste(gt_manip_roi, (x - roi_size, y - roi_size, x + roi_size + 1, y + roi_size + 1)[:2], gt_manip_roi.convert('RGBA'))

    # PIL to OpenCV conversion
    manip_img = np.array(manip_img.convert('RGB'))
    manip_img = manip_img[:, :, ::-1].copy()

    # Ground truth PIL to OpenCV conversion
    ground_truth = np.array(ground_truth.convert('RGB'))
    ground_truth = ground_truth[:, :, ::-1].copy()

    return manip_img, ground_truth


# Content-aware fill
def content_aware_fill(img, x, y, roi_size, method=cv2.INPAINT_TELEA):
    # Generate ground truth image
    ground_truth = get_ground_truth(img, x, y, roi_size)

    # Generate manipulated image
    manip_img = img.copy()
    manip_img = cv2.inpaint(manip_img, ground_truth, 3, flags=method)

    return manip_img, ground_truth


# Generate manipulated images
def main(args):
    # Welcome message
    print('Photo Forensics from Rounding Artifacts: manipulation script')
    print('Author: Paula Mihalcea')
    print('Version: 1.0')
    print('Based on a research by S. Agarwal and H. Farid. Details & source code at https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts.')
    print()

    start = time.time()

    # Get file list
    file_list = get_file_list(args.dir_path)

    progress_bar = tqdm.tqdm(total=len(file_list)*4*4*5)

    # Parameters
    manipulations = ['copy-move', 'median-filter', 'rotation', 'content-aware-fill']  # Manipulation type
    roi_sizes = [512, 256, 128, 64]  # Manipulated region (ROI) size
    jpeg_qualities = [(60, 70), (71, 80), (81, 90), (91, 100)]

    # Generate image
    for file in file_list:
        # Parameters
        img_path = args.dir_path + file
        filename, extension = get_filename(file)
        progress_bar.set_description('Processing image {}'.format(filename) + '.{}'.format(extension))

        # Create subfolders
        if not os.path.exists(args.dir_path + '/manip_jpeg/'):
            os.makedirs(args.dir_path + '/manip_jpeg/')
        if not os.path.exists(args.dir_path + '/manip_png/'):
            os.makedirs(args.dir_path + '/manip_png/')
        if not os.path.exists(args.dir_path + '/manip_jpeg/ground_truth/'):
            os.makedirs(args.dir_path + '/manip_jpeg/ground_truth/')
        if not os.path.exists(args.dir_path + '/manip_png/ground_truth/'):
            os.makedirs(args.dir_path + '/manip_png/ground_truth/')

        for manipulation in manipulations:
            for roi_size in roi_sizes:
                # Manipulation
                try:
                    manip_image, ground_truth = manipulate_image(img_path, manipulation, roi_size)  # Create a single manipulation of an image
                except (IOError, ValueError):
                    continue  # Ignore invalid images
                else:  # Save the manipulated image in 5 different formats...
                    for q in jpeg_qualities:  # ...JPEG with Q = [60, 70], JPEG with Q = [71, 80], JPEG with Q = [81, 90], JPEG with Q = [91, 100]...
                        quality = random.randint(q[0], q[1])  # For each range of JPEG qualities, a random JPEG quality is chosen in the specified range
                        cv2.imwrite(args.dir_path + '/manip_jpeg/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_{}'.format(quality) + '.jpeg', manip_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])  # JPEG
                        cv2.imwrite(args.dir_path + '/manip_jpeg/ground_truth/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_gt' + '.png', ground_truth)  # Ground truth (PNG only)

                cv2.imwrite(args.dir_path + '/manip_png/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '.png', manip_image)  # ...and PNG
                cv2.imwrite(args.dir_path + '/manip_png/ground_truth/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_gt' + '.png', ground_truth)  # Ground truth

                # Progress bar update
                progress_bar.update(5)

    end = time.time()

    # Final message
    print()
    print('Summary')
    print('Directory: {}'.format(args.dir_path))
    print('Images manipulated: {}.'.format(len(file_list)))
    print('Total generated images: {}.'.format(len(file_list)*4*4*5))
    if (end - start) / 60**2 >= 1:
        print('Elapsed time: {:.0f} h'.format((end - start) / 60**2) + ' {:.0f} m'.format((end - start) / 60) + ' {:.2f} s.'.format((end - start) % 60))
    elif (end - start) / 60 >= 1:
        print('Elapsed time: {:.0f} m'.format((end - start) / 60) + ' {:.2f} s.'.format((end - start) % 60))
    else:
        print('Elapsed time: {:.2f} s.'.format(end - start))
    print('Done.')

    return


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Manipulation script for the "Photo Forensics from Rounding Artifacts" project; generates manipulated images from a given directory (e.g. "path/to/images/") in two specific subdirectories (e.g. "path/to/images/manip_jpeg" and "path/to/images/manip_png") as described in the referenced paper.')

    # Add parser arguments
    parser.add_argument('dir_path', help='Path of the directory containing the images to be manipulated.')

    args = parser.parse_args()

    # Ensure folder path ends with "/"
    if args.dir_path[-1] != '/':
        args.dir_path += '/'

    # Run main script
    main(args)
