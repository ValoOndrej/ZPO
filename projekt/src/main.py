import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

def detect_circles_with_hough(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img = cv2.medianBlur(img, 5)
    rows = img.shape[0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    return cimg

def detect_center_of_gravity(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img = cv2.GaussianBlur(img, (5, 5), 0)
    rows = img.shape[0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)

    if circles is None:
        return None

    total_area = 0
    weighted_sum_x, weighted_sum_y = 0, 0
    for circle in circles[0]:
        area = np.pi * circle[2] ** 2
        total_area += area
        weighted_sum_x += circle[0] * area
        weighted_sum_y += circle[1] * area

    center_of_gravity_x = int(weighted_sum_x / total_area)
    center_of_gravity_y = int(weighted_sum_y / total_area)
    radius = int(np.sqrt(total_area/np.pi))
    cv2.circle(cimg, (center_of_gravity_x, center_of_gravity_y), 2, (0, 0 ,255), 3)
    cv2.circle(cimg, (center_of_gravity_x, center_of_gravity_y), radius, (0, 255, 0), 2)
    return cimg

def detect_largest_circle(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rows = img.shape[0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    largest_circle = max(circles, key=lambda x: x[2])
    cv2.circle(cimg, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 3)
    cv2.circle(cimg, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255 ,0), 3)
    return cimg

def detect_circles_with_Fourier(img_file):
    # Read the image and convert it to grayscale
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with maximum area
    contour = max(contours, key=cv2.contourArea)

    # Find the minimum enclosing circle of the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))

    # Apply Fourier transform to the grayscale image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Create a circular mask using the radius of the enclosing circle
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    D = np.sqrt((np.arange(rows)[:, np.newaxis] - crow) ** 2 + (np.arange(cols)[np.newaxis, :] - ccol) ** 2)
    mask = np.zeros((rows, cols), np.uint8)
    mask[D < radius] = 1
    fshift *= mask

    # Apply inverse Fourier transform to the masked spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Draw circles on the original image to mark the detected circle
    cv2.circle(img, center, 2, (0, 0, 255), 3)
    cv2.circle(img, center, int(radius), (0, 255, 0), 3)

    return img

def detect_circles_with_Centroid(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(largest_contour)
    if moments['m00'] == 0: 
        return None
    
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])

    radius = int(np.sqrt(moments["m00"] / np.pi))
    center = (centroid_x, centroid_y)
    cv2.circle(cimg, center, 2, (0, 0, 255), 3)
    cv2.circle(cimg, center, radius, (0, 255, 0), 2)

    return cimg

def detect_circles_with_HoughCentroid(img_file):
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    rows = gray_blur.shape[0]
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")
    center_x, center_y, total_pixels = 0, 0, 0
    for (x, y, r) in circles:
        moments = cv2.moments(gray_blur[y-r:y+r, x-r:x+r])
        if moments['m00'] == 0:
            continue
        center_x += int(x + moments["m10"] / moments["m00"])
        center_y += int(y + moments["m01"] / moments["m00"])
        total_pixels += 1

    center_x /= total_pixels
    center_y /= total_pixels
    radius = int(np.sqrt(moments["m00"] / np.pi))

    cv2.circle(img, (int(center_x), int(center_y)), 2, (0, 0, 255), 3)
    cv2.circle(img, (int(center_x), int(center_y)), radius, (0, 255, 0), 3)

    return img
    
def resize_img(img_file):
    if img_file is None:
        return img_file
    img = img_file

    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


if __name__ == "__main__":
    directory = 'Circle_Center_Detection'

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            img_file = os.path.join(subdir, file)
            
            img1 = resize_img(detect_circles_with_hough(img_file))
            if img1 is None: img1 = resize_img(cv2.imread(img_file))
            
            img2 = resize_img(detect_center_of_gravity(img_file))
            if img2 is None: img2 = resize_img(cv2.imread(img_file))
            
            img3 = resize_img(detect_largest_circle(img_file))
            if img3 is None: img3 = resize_img(cv2.imread(img_file))
            
            img4 = resize_img(detect_circles_with_Fourier(img_file))
            if img4 is None: img4 = resize_img(cv2.imread(img_file))
        
            img5 = resize_img(detect_circles_with_Centroid(img_file))
            if img5 is None: img5 = resize_img(cv2.imread(img_file))
            
            img6 = resize_img(detect_circles_with_HoughCentroid(img_file))
            if img6 is None: img6 = resize_img(cv2.imread(img_file))

            images = np.concatenate((img1, img2, img3, img4, img5, img6), axis=1)

            cv2.imshow(img_file, images)
            cv2.waitKey()
            cv2.destroyAllWindows()