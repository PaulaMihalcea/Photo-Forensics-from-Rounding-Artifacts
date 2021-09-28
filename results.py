import cv2
import matplotlib.pyplot as plt
import numpy as np
from manipulation import median_filter

img_path = 'ciao'
img_median = median_filter(img_path, 3)
