# Photo Forensics from Rounding Artifacts
## Author: Paula Mihalcea*
#### Università degli Studi di Firenze

---

![](https://img.shields.io/github/repo-size/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts)

Many aspects of **JPEG compression** have been successfully employed in
the domain of photo forensics. In particular, artifacts introduced by the choice of the rounding operator used to quantize the DCT coefficients can be used to localize tampering and identify specific encoders.

Following the research in [\[1\]](https://doi.org/10.1145/3369412.3395059), this work aims to provide a Python implementation of an **expectation maximization (EM) algorithm** to **localize inconsistencies** in these artifacts that arise from a variety of **image manipulations**. The resulting output map is computed as described in [\[2\]](https://doi.org/10.1109/WIFS.2017.8267641). Tests and results generated using 43 photos from a Canon EOS 5D Mark II camera, kindly provided by Andrea Mancini [\[3\]](https://www.biso.it/).

*Based on a research by S. Agarwal and H. Farid (see [\[1\]](https://doi.org/10.1145/3369412.3395059)).

## Contents
1. [Installation](#installation)
    - [Requirements](#requirements)
    - [Testing](#testing)
2. [Usage](#usage)
    - [Main](#main)
    - [Manipulation](#manipulation)
    - [Results](#results)
3. [Bibliography](#bibliography)
4. [License](#license)

## Installation

As a Python 3 application, this project has a few basic requirements in order to be up and running. In order to install them, the [`pip`](https://packaging.python.org/key_projects/#pip "pip") package installer is recommended, as it allows for the automatic installation of all requirements. Nonetheless, the latter have been listed in order to simplify an eventual manual installation.

It is assumed that Python 3 is already installed on the desired system.

1. Download the repository and navigate to its folder.

2. Install the requirements using `pip` from a terminal:

    ```
    pip install --upgrade -r requirements.txt
    ```

### Requirements

The following Python packages are required in order to run this program. Please note that unless otherwise specified, versions are to be intended as minimum.

| Package | Version |
| :------------ | :------------ |
| [Python](https://www.python.org/) | 3.8 |
| [argparse](https://docs.python.org/3/library/argparse.html) | _latest_ |
| [decimal](https://docs.python.org/3/library/decimal.html) | _latest_ |
| [Matplotlib](https://matplotlib.org/) | _latest_ |
| [NumPy](https://numpy.org/) | _latest_ |
| [OpenCV](https://opencv.org/) | _latest_ |
| [os](https://docs.python.org/3/library/os.html) | _latest_ |
| [Pillow](https://pillow.readthedocs.io/en/stable/) | _latest_ |
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
- `--win_size`: window size in pixel (default: `256`). Note: must be a multiple of 8.
- `--stop_threshold`: expectation-maximization algorithm stop threshold (default: `1e-2`);
- `--prob_r_b_in_c1`: expectation-maximization algorithm probability of _r_ conditioned by _b_ belonging to _C<sub>1</sub>_ (default: `0.3`);
- `--interpolate`: interpolate missing pixel values, aka NaNs generated from divisions in the EM algorithm, using the function from [\[4\]](https://stackoverflow.com/a/68558547), otherwise replace them with `0.5` (default: `False`). _Warning: slows down the program significantly_;
- `--sh`: show the resulting output map (default: `True`);
- `--sv`: save the resulting output map in the `results` folder (default: `False`);
- `--sh_diff_plot`: show the plot of the difference between successive estimates of template _c_ (default: `False`);
- `--sv_diff_plot`: save the plot of the difference between successive estimates of template _c_ in the `results` folder (default: `False`).

Example call with optional arguments:
```
python3 main.py "images/my_photo.jpg" --win_size=256 --stop_threshold=1e-2 --save=True
```

### Manipulation

This script generates **manipulated images** and their respective **ground truth masks** from a given directory (`path/to/images/`) in three specific subdirectories (`path/to/images/manip_jpeg`, `path/to/images/manip_png` and `path/to/images/manip_gt`), as described in [\[1\]](https://doi.org/10.1145/3369412.3395059).

Specifically, for every original image the script generates **80 manipulated images** (ground truth masks excluded), one for each:

- manipulation type:
  - **copy-move**;
  - **median filter**: 3x3 OpenCV median filter;
  - **rotation**: random rotation of 10 to 80 degrees;
  - **content-aware fill**: OpenCV `inpaint()` function [\[5\]](https://docs.opencv.org/4.5.2/d7/d8b/group__photo__inpaint.html) with Telea method [\[6\]](https://doi.org/10.1080/10867651.2004.10487596);
- **region size**: 512 px, 256 px, 128 px and 64 px;
- **JPEG quality**: a random quality chosen from each of the ranges \[60, 70\], \[71, 80\], \[81, 90\] and \[91, 100\];
- **save format**: PNG and JPEG (OpenCV `imwrite()` function [\[7\]](https://docs.opencv.org/4.5.2/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)).

The script can be run with:

```
python3 manipulate.py "path/to/images/"
```

### Results

TODO

## Bibliography
[\[1\]](https://doi.org/10.1145/3369412.3395059) Shruti Agarwal and Hany Farid. 2020. **Photo Forensics From Rounding Artifacts.** In Proceedings of the 2020 ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '20). Association for Computing Machinery, New York, NY, USA, 103–114, DOI:[10.1145/3369412.3395059](https://doi.org/10.1145/3369412.3395059)

[\[2\]](https://doi.org/10.1109/WIFS.2017.8267641) Shruti Agarwal and Hany Farid. 2017. **Photo Forensics from JPEG Dimples.** 2017 IEEE Workshop on Information Forensics and Security (WIFS), pp. 1-6, DOI:[10.1109/WIFS.2017.8267641](https://doi.org/10.1109/WIFS.2017.8267641)

[\[3\]](https://www.biso.it/) Andrea Mancini, UX Designer fixed with www -  [biso.it](https://www.biso.it/)

[\[4\]](https://stackoverflow.com/a/68558547) Sam De Meyer, **[interpolate missing values 2d python](https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python)**, 2021

[\[5\]](https://docs.opencv.org/4.5.2/d7/d8b/group__photo__inpaint.html) OpenCV, **Inpainting**, OpenCV Documentation

[\[6\]](https://doi.org/10.1080/10867651.2004.10487596) Alexandru Telea, **An image inpainting technique based on the fast marching method**, Journal of graphics tools, 9(1):23–34, 2004, DOI:[10.1080/10867651.2004.10487596](https://doi.org/10.1080/10867651.2004.10487596)

[\[7\]](https://docs.opencv.org/4.5.2/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) OpenCV, **imwrite()**, OpenCV Documentation

## License
This work is licensed under a [Creative Commons “Attribution-NonCommercial-ShareAlike 4.0 International”](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license. More details are available in the [LICENSE](./LICENSE) file. All rights reserved to the original paper's authors. 
