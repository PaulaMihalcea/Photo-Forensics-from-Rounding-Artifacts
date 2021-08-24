# Photo Forensics from Rounding Artifacts
## Author: Paula Mihalcea*
#### Università degli Studi di Firenze

*Based on a research by S. Agarwal and H. Farid (see [\[1\]](https://doi.org/10.1145/3369412.3395059)).

---

![](https://img.shields.io/github/repo-size/PaulaMihalcea/Photo-Forensics-from-Rounding-Artifacts)

Many aspects of **JPEG compression** have been successfully used in
the domain of photo forensics. In particular, artifacts introduced by the choice of rounding operator used to quantize the DCT coefficients can be used to localize tampering and identify specific encoders.

Following the research in [\[1\]](https://doi.org/10.1145/3369412.3395059), this work aims to provide a Python implementation of an **algorithm** to **localize inconsistencies** in these artifacts that arise from a variety of **image manipulations**.

## Usage

(TODO)

### Requirements

The following Python packages are needed in order to run this project:

- [`argparse`](https://docs.python.org/3/library/argparse.html)
- [`numpy`](https://numpy.org/)
- [`opencv-python`](https://docs.opencv.org/4.5.2/index.html)

## Bibliography
[\[1\]](https://doi.org/10.1145/3369412.3395059) Shruti Agarwal and Hany Farid. 2020. Photo Forensics From Rounding Artifacts. In Proceedings of the 2020 ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec '20). Association for Computing Machinery, New York, NY, USA, 103–114. DOI:[https://doi.org/10.1145/3369412.3395059](https://doi.org/10.1145/3369412.3395059)

## License
This work is licensed under a [Creative Commons “Attribution-NonCommercial-ShareAlike 4.0 International”](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license. More details are available in the [LICENSE](./LICENSE) file.
