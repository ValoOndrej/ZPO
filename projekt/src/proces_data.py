import os
import cv2
import sys
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure

def detect_circles_with_hough(img, hough_img=None):
    img = img.copy()
    h_img = hough_img
    if h_img is None:
        h_img = img
    h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY)

    # img = cv2.medianBlur(img, 5)
    rows = h_img.shape[0]
    circles = cv2.HoughCircles(h_img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)
    if circles is None:
        return img
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img


def average_surroundings(image, col, row):
    new_image = image.copy()
    neighbours = new_image[col-1:col+2, row-1:row+2, :]
    neighbours = neighbours.reshape((-1, 3))
    neighbours = neighbours[(neighbours[:, 0] != 0) | (neighbours[:, 1] != 0)]
    return tuple(np.round(np.mean(neighbours, axis=0)).astype(int))


def remove_cross(image):
    new_image = np.copy(image)
    for col in range(image.shape[0]):
        for row in range(image.shape[1]):
            pixel = image[col, row]
            if not np.all(pixel == pixel[0]):
                new_image[col, row] = average_surroundings(new_image, col, row)
    return new_image


def apply_hog(image):
    new_image = image.copy()
    fd, hog_image = hog(new_image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_rescaled_with_dim = np.expand_dims(hog_image_rescaled, axis=-1)  # add third dimension
    hog_image_rescaled_with_dim = np.repeat(hog_image_rescaled_with_dim, 3, axis=-1)
    hog_image_rescaled = hog_image_rescaled_with_dim * 255.0

    # Convert the pixel values to integers
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    return hog_image_rescaled
    
def resize_img(img, scale_percent=50):
    new_image = img.copy()
    width = int(new_image.shape[1] * scale_percent / 100)
    height = int(new_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)

np.set_printoptions(threshold=sys.maxsize)
if __name__ == '__main__':
    directory = 'Circle_Center_Detection'
    img_file = 'Circle_Center_Detection/complicated_detection/2022-07-21_14-12-31-361_InputImageWithCenterCross_.png'
    #img_file = 'Circle_Center_Detection/easy_to_detect/30.00kV_0.80nA_2.07mm_620.png'

    """
    for subdir, dirs, files in os.walk(directory):
        for file_name in files:
            img_file = os.path.join(subdir, file_name)
    """

    img = resize_img(cv2.imread(img_file))
    no_cross =  remove_cross(img)
    hog_img = apply_hog(no_cross)
    hough = detect_circles_with_hough(no_cross)
    hog_hough = detect_circles_with_hough(hog_img)
    img_hough = detect_circles_with_hough(no_cross, hog_img)

    """
    print('Shape of img:', img.shape)
    print('Shape of no_cross:', no_cross.shape)
    print('Shape of hog_img:', hog_img.shape)
    print('Shape of hough:', hough.shape)
    print('Shape of hog_hough:', hog_hough.shape)
    print('Shape of img_hough:', img_hough.shape)
    """

    images = np.concatenate((img, no_cross, hog_img, hough, hog_hough, img_hough), axis=1)

    cv2.imshow(img_file,images)
    cv2.waitKey()
    cv2.destroyAllWindows()