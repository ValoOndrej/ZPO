# ZPO Project
xvalo00

## Installation

You can install all the requirements using `pip`.

### PIP

```
$ pip3 install -r requirements.txt
```
## Usage

### Expected Directory Structure

```
--  |-- src
    |-- Circle_Center_Detection
        |-- komplikovaná_detekcia
        |-- ľahko_zistiteľné
        |-- nevidno
        |-- na_okraji
```

### Execution
It is expected that the script will be run from the project's root directory.
```
$ src/process_data.py -s
```
- `-s` displays images with the detected center.

# ZPO - Tasks

## Task 1
- **Task 1.1**: `rgb2gray` converts an RGB image to grayscale by averaging RGB values.
- **Task 1.2**: `convolution` applies a 3x3 kernel to a grayscale image, zeroing border values.

Execution:
```
./image_processing <path_to_image>
```

## Task 2
- **Task 2.1**: `geometricalTransform` applies geometric transformations with nearest neighbor interpolation.
Execution:
```
./mt02 <image_path> [rotation in degrees] [scale]
```

## Task 3
- **Task 3.1**: `passFilter` uses high-pass or low-pass filters in the frequency domain.
Execution:
```
./mt03 <image_path> <spatial_frequency_limit> [path to reference results]
```

## Task 4
- **Task 4.1**: `noiseSaltAndPepper` adds salt and pepper noise.
- **Task 4.2**: `noiseGaussian` adds Gaussian noise.
Execution:
```
./mt04 <image_path> <noise_type> <param_1> [param_2 = 0.0]
```
- `noise_type`: sp (salt & pepper) or gn (Gaussian).
- `param_1`: probability for sp or standard deviation for gn.
- `param_2`: mean for Gaussian noise (default is 0.0).

## Task 5
- **Task 5.0**: `getPSNR` calculates Peak Signal-to-Noise Ratio (PSNR) between two images.
- **Task 5.1**: `medianFilter` applies a median filter to remove noise.
- **Task 5.2**: `gaussianFilter` blurs an image using a Gaussian filter.
Execution:
```
./mt05 <image_path> <filter_type> <size> [sigma = 0.0]
```
- `filter_type`: mf (median filter) or gf (Gaussian filter).
- `size`: filter size.
- `sigma`: standard deviation for Gaussian filter (default is 1.0).
