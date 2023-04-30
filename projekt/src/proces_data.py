import os
import cv2
import sys
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure
from scipy import ndimage

def detect_circles_with_hough(img, hough_img=None):
    img = img.copy()
    h_img = hough_img
    if h_img is None:
        h_img = img
    h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(h_img, cv2.HOUGH_GRADIENT, 1, 10,
                              param1=30, param2=90, minRadius=2, maxRadius=80)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
        cv2.circle(img, (i[0], i[1]), 0, (0, 0, 255), 2)
    return img

def average_surroundings(img, col, row):
    new_b, new_g, new_r = 0, 0, 0
    cnt = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (col + i < 0 or col + i >= img.shape[0] or 
                row + j < 0 or row + j >= img.shape[1]):
                # skip pixels outside the image boundaries
                continue
            if not np.all(img[col+i, row+j] == img[col+i, row+j][0]):
                continue
            cnt += 1
            new_b += img[col+i, row+j][0]
            new_g += img[col+i, row+j][1]
            new_r += img[col+i, row+j][2]
    return (int(new_b/cnt), int(new_g/cnt), int(new_r/cnt))



def remove_cross(image):
    new_image = np.copy(image)
    for col in range(new_image.shape[0]):
        for row in range(new_image.shape[1]):
            pixel = new_image[col, row]
            if not np.all(pixel == pixel[0]):
                new_image[col, row] = average_surroundings(new_image, col, row)
    return new_image


def apply_hog(image, pixels=(32, 32)):
    new_image = image.copy()
    fd, hog_image = hog(new_image, orientations=16, pixels_per_cell=pixels,
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_rescaled_with_dim = np.expand_dims(hog_image_rescaled, axis=-1)  # add third dimension
    hog_image_rescaled_with_dim = np.repeat(hog_image_rescaled_with_dim, 3, axis=-1)
    hog_image_rescaled = hog_image_rescaled_with_dim * 255.0

    # Convert the pixel values to integers
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    return hog_image_rescaled

def multiply(image):
    new_image = np.copy(image)
    for col in range(new_image.shape[0]):
        for row in range(new_image.shape[1]):
            if new_image[col, row][0] <= 10:
                new_image[col, row] = (0, 0, 0)
            else:
                new_image[col, row] = (new_image[col, row] *2)
    return new_image

def resize_img(img, scale_percent=100):
    new_image = img.copy()
    width = int(new_image.shape[1] * scale_percent / 100)
    height = int(new_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)

def show(name,img):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 10,40)
    cv2.imshow(name,img)


np.set_printoptions(threshold=sys.maxsize)
if __name__ == '__main__':
    directory = 'Circle_Center_Detection'
    img_file = 'Circle_Center_Detection/complicated_detection/2022-01-26_16-05-05-931_Step1SpiralFindRoughCenter_wd_4.0000E-003_hfw_1.00E-003.png'
    # img_file = 'Circle_Center_Detection/easy_to_detect/30.00kV_0.80nA_2.07mm_624.png'

    
    for subdir, dirs, files in os.walk(directory):
        for file_name in files:
            img_file = os.path.join(subdir, file_name)
    

            img = resize_img(cv2.imread(img_file))
            no_cross =  remove_cross(img)
            hog_48 = apply_hog(no_cross, pixels=(48, 48))
            hog_32 = apply_hog(no_cross, pixels=(32, 32))
            hog_24 = apply_hog(no_cross, pixels=(24, 24))
            hog_16 = apply_hog(no_cross, pixels=(16, 16))
            hog_12 = apply_hog(no_cross, pixels=(12, 12))
            hog_8 = apply_hog(no_cross, pixels=(8, 8))

            hog_img = cv2.add(cv2.add(hog_48, hog_32, hog_24), cv2.add(hog_16, hog_12, hog_8))

            hough = detect_circles_with_hough(no_cross, cv2.subtract(no_cross, hog_img))
            if hough is not None:
                images = np.concatenate((hog_img, hough), axis=1)
                show(img_file, images)
                continue

            print(f"{img_file}: not found")

        cv2.waitKey()
        cv2.destroyAllWindows()
        print("==============================")