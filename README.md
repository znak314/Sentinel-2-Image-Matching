# Task 2. Sentinel-2 image matching
In this task I worked on the algorithm for matching satellite images
## Overview of the solution
Data for this task was extracted from the [dataset](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine). Each image is read in grayscale mode, reducing the complexity of the data by eliminating color information. The images are resized to ensure that their maximum dimension (width or height) does not exceed a size of 1024 pixels. The processed images are saved as `JPG` files in the output directory, using a 90% compression. Image matching was performed by using SIFT.

## SIFT Feature Matching Algorithm

The [SIFT](https://towardsdatascience.com/sift-scale-invariant-feature-transform-c7233dc60f37) (Scale-Invariant Feature Transform) algorithm is a robust feature extraction method used in computer vision to identify and describe local features in images, it is invariant to scaling, rotation, and changes in illumination.

### Steps in the SIFT Feature Matching Algorithm:

1. **Feature Detection**:
   - SIFT identifies keypoints in images that are stable across different scales and orientations.

2. **Descriptor Calculation**:
   - For each detected keypoint, a descriptor is computed, which is a vector representing the local gradient information around the keypoint.

3. **Feature Matching**:
   - The algorithm matches features from two images using a brute-force matcher (`cv2.BFMatcher`). The descriptors of the two images are compared to find the best matches based on a L2 norm metric.

4. **Fundamental Matrix Estimation**:
   - The fundamental matrix is calculated using RANSAC (Random Sample Consensus) to identify the geometric relationship between the two sets of matched keypoints.

### Possible issues
The algorithm does not work very accurately at different times of the year, but it is very accurate at the same seasons.

## Project structure
* `dataset_creation.ipynb` - a notebook with preprocessing of input dataset
* `algorithm.py` -  python script with SIFT matching implementation. 
* `model_inference.py` -  python script for performing images matching
* `demo_notebook.ipynb` - notebook with a demonstration of work of the algorithm 
* `images for demo` - folder with some images used in demonstration 
* `Boost ideas.pdf` - pdf with a description of possible improvements

## How to set-up a project?
### 1. **Clone the repository**
   Clone this repository to your local machine using:

   ```bash
   git clone git clone https://github.com/znak314/Sentinel-2-Image-Matching.git
   ```
### 2. **Install all necessary libraries**
   Install all dependencies by using:

   ```bash
   pip install -r requirements.txt
   ```
### 3. **Run the script**
   You can download more images for algorithm testing from [google drive](https://drive.google.com/drive/folders/1VvZiU7yqYirCkhieWG5h2kphKGUA6jDY?dmr=1&ec=wgc-drive-globalnav-goto) or use images from `images for demo` by default.

```
python model_inference.py 
```
