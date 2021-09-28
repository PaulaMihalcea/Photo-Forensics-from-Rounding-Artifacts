import cv2
import numpy as np
import os
import random
import time
import tqdm
from argparse import ArgumentParser
from PIL import Image
from utils import get_filename, get_image_size, load_image


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
    if args.dir_path[-1] != '/':
        args.dir_path += '/'

    valid_extensions = ['.jpeg', '.jpg', '.jpe', '.jfif', '.jif', '.png']
    file_list = [img_file for img_file in os.listdir(args.dir_path) for ext in valid_extensions if img_file.endswith(ext)]
    progress_bar = tqdm.tqdm(total=len(file_list)*4*4*5)

    # Parameters
    manipulations = ['copy_move', 'median_filter', 'rotation', 'content_aware_fill']  # Manipulation type
    roi_sizes = [512, 256, 128, 64]  # Manipulated region (ROI) size
    jpeg_qualities = [(60, 70), (71, 80), (81, 90), (91, 100)]

    # Generate image
    for file in file_list:
        # Parameters
        img_path = args.dir_path + file
        filename, extension = get_filename(img_path)
        progress_bar.set_description('Processing image {}'.format(filename) + '.{}'.format(extension))

        # Create subfolders
        if not os.path.exists(args.dir_path + '/manip_jpeg/'):
            os.makedirs(args.dir_path + '/manip_jpeg/')
        if not os.path.exists(args.dir_path + '/manip_png/'):
            os.makedirs(args.dir_path + '/manip_png/')
        if not os.path.exists(args.dir_path + '/manip_gt/'):
            os.makedirs(args.dir_path + '/manip_gt/')

        for manipulation in manipulations:
            for roi_size in roi_sizes:
                # Manipulation
                try:
                    manip_image, ground_truth = manipulate_image(img_path, manipulation, roi_size)  # Create a single manipulation of an image
                except IOError:
                    continue  # Ignore invalid images
                else:  # Save the manipulated image in 5 different formats...
                    for q in jpeg_qualities:  # ...JPEG with Q = [60, 70], JPEG with Q = [71, 80], JPEG with Q = [81, 90], JPEG with Q = [91, 100]...
                        quality = random.randint(q[0], q[1])  # For each range of JPEG qualities, a random JPEG quality is chosen in the specified range
                        cv2.imwrite(args.dir_path + '/manip_jpeg/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_{}'.format(quality) + '.jpeg', manip_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])  # JPEG
                        cv2.imwrite(args.dir_path + '/manip_jpeg/manip_gt/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_gt' + '.png', ground_truth)  # Ground truth (PNG only)

                cv2.imwrite(args.dir_path + '/manip_png/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '.png', manip_image)  # ...and PNG
                cv2.imwrite(args.dir_path + '/manip_png/manip_gt/' + filename + '_' + manipulation + '_{}'.format(roi_size) + '_gt' + '.png', ground_truth)  # Ground truth

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

    return


# Define random manipulation ROI
def manipulate_image(img_path, manipulation, roi_size):
    # Load image
    img = load_image(img_path)
    img_height, img_width = get_image_size(img)

    # Randomly choose manipulation ROI position
    x = random.randint(0, img_height - roi_size)
    y = random.randint(0, img_width - roi_size)

    roi = (y, x, roi_size)

    # Ground truth image
    ground_truth = img.copy().astype(float)
    ground_truth[y:y + roi_size, x:x + roi_size] = np.inf
    ground_truth = np.where(ground_truth == np.inf, 255, 0).astype(np.uint8)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    # Apply manipulation
    # Copy-move
    if manipulation == 'copy_move':
        manip_img = copy_move(img, roi)

    # Median filter
    elif manipulation == 'median_filter':
        manip_img = median_filter(img, roi)

    # Rotation
    elif manipulation == 'rotation':
        manip_img = rotate(img_path, roi)

    # Content-aware fill
    elif manipulation == 'content_aware_fill':
        manip_img = content_aware_fill(img, ground_truth)

    else:
        raise ValueError('Invalid manipulation method. Possible values: "copy_move", "median_filter", "rotation", "content_aware_fill".')

    return manip_img, ground_truth


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

    return manip_img


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

    return manip_img


# Rotation
def rotate(img_path, roi):
    # Load image (PIL)
    img = Image.open(img_path).convert('RGB')

    # Define ROI parameters
    x = roi[0]
    y = roi[1]
    roi_size = roi[2]

    # Randomly choose rotation degrees
    degrees = random.randint(10, 80)

    # Manipulate ROI
    manip_roi = img.crop(box=(x, y, x + roi_size, y + roi_size))
    manip_roi.rotate(degrees, expand=1)
    manip_roi.resize((roi_size, roi_size))

    # Generate manipulated image
    manip_img = img.copy()
    manip_img.paste(manip_roi, (x, y, x + roi_size, y + roi_size), manip_roi.convert('RGBA'))

    # PIL to OpenCV conversion
    manip_img = np.array(manip_img)
    manip_img = manip_img[:, :, ::-1].copy()

    return manip_img


# Content-aware fill
def content_aware_fill(img, ground_truth, method=cv2.INPAINT_TELEA):
    # Generate manipulated image
    manip_img = img.copy()
    manip_img = cv2.inpaint(manip_img, ground_truth, 3, flags=method)

    return manip_img


if __name__ == '__main__':

    # Initialize parser
    parser = ArgumentParser(description='Manipulation script for the "Photo Forensics from Rounding Artifacts" project; generates manipulated images from a given directory (e.g. "path/to/images/") in a specific subdirectory (e.g. "path/to/images/manip") as described in the referenced paper.')

    # Add parser arguments
    parser.add_argument('dir_path', help='Path of the directory containing the images to be manipulated.')

    args = parser.parse_args()

    # Run main script
    main(args)
