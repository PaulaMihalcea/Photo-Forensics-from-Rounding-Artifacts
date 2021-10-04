# Photo Forensics from Rounding Artifacts: a Python implementation
## Author: Paula Mihalcea
#### Università degli Studi di Firenze

---

![](https://img.shields.io/github/repo-size/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts)

Many aspects of **JPEG compression** have been successfully employed in
the domain of photo forensics. In particular, artifacts introduced by the choice of the rounding operator used to quantize the DCT coefficients can be used to localize tampering and identify specific encoders.

Following the research in [\[1\]](https://doi.org/10.1145/3369412.3395059), this work aims to provide a Python implementation of an **expectation maximization (EM) algorithm** to **localize inconsistencies** in these artifacts that arise from a variety of **image manipulations**. The resulting output map is computed as described in [\[2\]](https://doi.org/10.1109/WIFS.2017.8267641).

Based on a **research by S. Agarwal and H. Farid** (see [\[1\]](https://doi.org/10.1145/3369412.3395059)). Results generated using a dataset kindly provided by **ing. Marco Fontani** (Amped Software) through **prof. Alessandro Piva** (Università degli Studi di Firenze).

## Contents
1. [GUI](#gui)
2. [Installation](#installation)
    - [Requirements](#requirements)
    - [Testing](#testing)
3. [Usage](#usage)
   - [Main](#main)
   - [Manipulation](#manipulation)
   - [Results](#results)
   - [Amped report parsing script](#amped-report-parsing-script)
4. [Bibliography](#bibliography)
5. [License](#license)

## GUI

This algorithm has been added as a feature to **IEViewer**, a simple Python image viewer, providing a **neat graphical interface** to an otherwise nonintuitive script; in this version it employs the default settings, and does not use ground truth maps. Check it out before cloning this repository if you are only interested in a basic usage.
   <p align="center"><img src="https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts/blob/master/screenshots/analyze_0.png" width="50%" height="50%"></p>
    <p align="center"><img src="https://github.com/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts/blob/master/screenshots/analyze_1.png" width="50%" height="50%"></p>

## Installation

As a Python 3 application, this project has a few basic requirements in order to be up and running. In order to install them, the [`pip`](https://packaging.python.org/key_projects/#pip "pip") package installer is recommended, as it allows for the automatic installation of all requirements. Nonetheless, the latter have been listed in order to simplify an eventual manual installation.

It is assumed that Python 3 is already installed on the desired system.

1. Download the repository and navigate to its folder.

2. Install the requirements using `pip` from a terminal:

    ```
    pip install --upgrade -r requirements.txt
    ```

### Requirements

The following Python packages are required in order to run this program. Please note that versions are to be intended as minimum, or the latest compatible.

| Package | Version |
| :------------ | :------------ |
| [Python](https://www.python.org/) | 3.8 |
| [argparse](https://docs.python.org/3/library/argparse.html) | _latest_ |
| [decimal](https://docs.python.org/3/library/decimal.html) | _latest_ |
| [Matplotlib](https://matplotlib.org/) | _latest_ |
| [NumPy](https://numpy.org/) | _latest_ |
| [OpenCV](https://opencv.org/) | _latest_ |
| [os](https://docs.python.org/3/library/os.html) | _latest_ |
| [pandas](https://pandas.pydata.org/) | _latest_ |
| [Pillow](https://pillow.readthedocs.io/en/stable/) | 8.2.0 |
| [random](https://docs.python.org/3/library/random.html) | _latest_ |
| [scikit-learn](https://scikit-learn.org/stable/) | _latest_ |
| [SciPy](https://www.scipy.org/) | _latest_ |
| [sys](https://docs.python.org/3/library/sys.html) | _latest_ |
| [time](https://docs.python.org/3/library/time.html) | _latest_ |
| [tqdm](https://github.com/tqdm/tqdm) | _latest_ |

### Testing
This project has been written and tested using [Python 3.8](https://www.python.org/downloads/release/python-380/) on a Windows 10 Pro machine.

## Usage

### Main

Run from a terminal specifying the path to the image to be analyzed, as follows:

```
python3 main.py "path/to/image/image_file.jpg"
```

Optional arguments:
- `--win_size`: window size in pixel (default: `64`), must be a multiple of 8;
- `--stop_threshold`: expectation-maximization algorithm stop threshold (default: `1e-3`);
- `--prob_r_b_in_c1`: expectation-maximization algorithm probability of _r_ conditioned by _b_ belonging to _C<sub>1</sub>_ (default: `0.5`);
- `--interpolate`: interpolate missing pixel values, aka NaNs generated from divisions in the EM algorithm, using the function from [\[3\]](https://stackoverflow.com/a/68558547), otherwise replace them with `0.5` (default: `False`). _Warning: slows down the program significantly_;
- `--show`: show the resulting output map (default: `True`);
- `--save`: save the resulting output map in the `results` folder (default: `False`);
- `--show_roc_plot`: show the plot of the ROC curve (default: `False`);
- `--save_roc_plot`: save the plot of the ROC curve in the `results` folder (default: `False`);
- `--show_diff_plot`: show the plot of the difference between successive estimates of template _c_ (default: `False`);
- `--save_diff_plot`: save the plot of the difference between successive estimates of template _c_ in the `results` folder (default: `False`);
- `--verbose`: show progress in terminal (default: `True`).

Example call with optional arguments:
```
python3 main.py "images/my_photo.jpg" --win_size=256 --stop_threshold=1e-2 --save=True
```

### Manipulation

This script generates **manipulated images** and their respective **ground truth masks** from a given directory (`path/to/images/`) in four subdirectories (`path/to/images/manip_jpeg`, `path/to/images/manip_png`, `path/to/images/manip_jpeg/ground_truth` and `path/to/images/manip_png/ground_truth`), as described in [\[1\]](https://doi.org/10.1145/3369412.3395059).

Specifically, for every original image the script generates **80 manipulated images** (ground truth masks excluded), one for each:

- manipulation type:
  - **copy-move**;
  - **median filter**: 3x3 OpenCV median filter;
  - **rotation**: random rotation of 10 to 80 degrees;
  - **content-aware fill**: OpenCV `inpaint()` function [\[4\]](https://docs.opencv.org/4.5.2/d7/d8b/group__photo__inpaint.html) with Telea method [\[5\]](https://doi.org/10.1080/10867651.2004.10487596);
- **region size**: 512 px, 256 px, 128 px and 64 px;
- **JPEG quality**: a random quality chosen from each of the ranges \[60, 70\], \[71, 80\], \[81, 90\] and \[91, 100\];
- **save format**: PNG and JPEG (OpenCV `imwrite()` function [\[6\]](https://docs.opencv.org/4.5.2/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)).

The script can be run with:

```
python3 manipulation.py "path/to/images/"
```

### Results

This script generates the plots shown in figures 6 and 7 of [\[1\]](https://doi.org/10.1145/3369412.3395059) (except for figure _7(d)_) using images manipulated as explained in the same paper. It can be used to either analyze images in a given directory and save the results (as CSV files and PNG images, respectively in a `results` and `plots` subfolder), or to create the plots from existing CSV files.

In order to analyze all images and generate results, the script can be run with:

```
python3 results.py True --dir_path="path/to/images/"
```

Optional arguments:
- `--win_size`: window size in pixel (default: `64`). Agarwal & Farid use `64`, `128` and `256`, for three different sets of experiments [\[1\]](https://doi.org/10.1145/3369412.3395059).

As mentioned, the script can also be used to create plots from existing results, assuming they have been generated with the previous command and exist as CSV files in the `results` subfolder:

```
python3 results.py False --res_path="results_file.csv"
```

All ROC curves in this project have been calculated with the function from [\[8\]](https://stackoverflow.com/a/61323665), in order to get a fixed number of thresholds and easily calculate the average ROC curve.

### Amped report parsing script

This script parses an Amped Authenticate HTML report [\[7\]](https://ampedsoftware.com/authenticate) containing information about the dimples' strength of an image dataset, and saves its contents to a CSV file (`results/report.csv`) for easier indexing.  Only selects images containing dimples stronger than 15 with offset [0, 0] are selected.

After the creation of the CSV report, the program can be used to randomly select _n_ images for each of three dimples strength ranges, in order to provide new dataset partitions for further data insight:
- **low dimple strength**: \[15, 30\];
- **medium dimple strength**: \[31, 45\];
- **high dimple strength**: >= 45.

**Note:** This is a highly situational script, and as such has not been optimized for command line execution: variables must be inserted manually into the code before execution. It has only been included for completeness' sake.

## Bibliography
[\[1\]](https://doi.org/10.1145/3369412.3395059) Shruti Agarwal and Hany Farid. 2020. **Photo Forensics From Rounding Artifacts.** In Proceedings of the 2020 ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '20). Association for Computing Machinery, New York, NY, USA, 103–114, DOI:[10.1145/3369412.3395059](https://doi.org/10.1145/3369412.3395059)

[\[2\]](https://doi.org/10.1109/WIFS.2017.8267641) Shruti Agarwal and Hany Farid. 2017. **Photo Forensics from JPEG Dimples.** 2017 IEEE Workshop on Information Forensics and Security (WIFS), pp. 1-6, DOI:[10.1109/WIFS.2017.8267641](https://doi.org/10.1109/WIFS.2017.8267641)

[\[3\]](https://stackoverflow.com/a/68558547) Sam De Meyer, **[interpolate missing values 2d python](https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python)**, 2021

[\[4\]](https://docs.opencv.org/4.5.2/d7/d8b/group__photo__inpaint.html) OpenCV, **Inpainting**, OpenCV Documentation

[\[5\]](https://doi.org/10.1080/10867651.2004.10487596) Alexandru Telea, **An image inpainting technique based on the fast marching method**, Journal of graphics tools, 9(1):23–34, 2004, DOI:[10.1080/10867651.2004.10487596](https://doi.org/10.1080/10867651.2004.10487596)

[\[6\]](https://docs.opencv.org/4.5.2/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) OpenCV, **imwrite()**, OpenCV Documentation

[\[7\]](https://ampedsoftware.com/authenticate) Amped Software, **Amped Authenticate**, 09.2021

[\[8\]](https://stackoverflow.com/a/61323665) Flavia Giammarino, **[How to calculate TPR and FPR in Python without using sklearn?](https://stackoverflow.com/a/61323665)**, 2020

## License
This work is licensed under a [Creative Commons “Attribution-NonCommercial-ShareAlike 4.0 International”](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license. More details are available in the [LICENSE](./LICENSE) file. All rights regarding the theory behind the EM algorithm reserved to the original paper's authors.
