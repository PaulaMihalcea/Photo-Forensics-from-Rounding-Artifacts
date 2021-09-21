# Photo Forensics from Rounding Artifacts
## Author: Paula Mihalcea*
#### Università degli Studi di Firenze

---

![](https://img.shields.io/github/repo-size/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts)

Many aspects of **JPEG compression** have been successfully used in
the domain of photo forensics. In particular, artifacts introduced by the choice of rounding operator used to quantize the DCT coefficients can be used to localize tampering and identify specific encoders.

Following the research in [\[1\]](https://doi.org/10.1145/3369412.3395059), this work aims to provide a Python implementation of an **algorithm** to **localize inconsistencies** in these artifacts that arise from a variety of **image manipulations**.

_<sup>*Based on a research by S. Agarwal and H. Farid (see [\[1\]](https://doi.org/10.1145/3369412.3395059)).</sup>_

## Usage

Run from any terminal specifying the path to the image to be analyzed, as follows:

```
python3 main.py "path/to/image/image_file.jpg"
```

Optional arguments:
- `--win_size`: window size in pixel (default: `256`). Note: must be a multiple of 8.
- `--stop_threshold`: expectation-maximization algorithm stop threshold (default: `1e-3`);

- `--prob_r_b_in_c1`: expectation-maximization algorithm probability of _r_ conditioned by _b_ belonging to _C<sub>1</sub>_ (default: `0.5`);
- `--interpolate`: interpolate missing pixel values, aka NaNs generated from divisions in the EM algorithm (default: `False`). _Warning: slows down the program significantly_;
- `--show`: show the resulting output map (default: `False`);
- `--save`: save the resulting output map in the `results` folder (default: `False`);
- `--show_diff_plot`: show the plot of the difference between successive estimates of template _c_ (default: `False`);
- `--save_diff_plot`: save the plot of the difference between successive estimates of template _c_ in the `results` folder (default: `False`).

Example call with optional arguments:
```
python3 "images/my_photo.jpg" --win_size=256 --stop_threshold=1e-2 --save=True
```

### Requirements

The following Python packages are needed in order to run this project:

- [`argparse`](https://docs.python.org/3/library/argparse.html)
- [`numpy`](https://numpy.org/)
- [`opencv-python`](https://docs.opencv.org/4.5.2/index.html)

This program has been written and tested using [`Python 3.8`](https://www.python.org/downloads/release/python-380/).

## Bibliography
[\[1\]](https://doi.org/10.1145/3369412.3395059) Shruti Agarwal and Hany Farid. 2020. **Photo Forensics From Rounding Artifacts.** In Proceedings of the 2020 ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '20). Association for Computing Machinery, New York, NY, USA, 103–114. DOI:[https://doi.org/10.1145/3369412.3395059](https://doi.org/10.1145/3369412.3395059)

## License
This work is licensed under a [Creative Commons “Attribution-NonCommercial-ShareAlike 4.0 International”](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license. More details are available in the [LICENSE](./LICENSE) file.
