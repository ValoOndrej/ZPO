import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from scipy.spatial.distance import cdist
import argparse

def detect_circles_with_hough(img, hough_img=None):
    img = img.copy()
    h_img = hough_img
    if h_img is None:
        h_img = img
    h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(h_img, cv2.HOUGH_GRADIENT, 1, 10,
                              param1=30, param2=90, minRadius=2, maxRadius=80)
    if circles is None:
        return None, [0, 0]
    centers = circles[0, :, :2]
    average_center = np.mean(centers, axis=0).astype(np.int16)
    radius = int(np.mean(circles[0, :, 2]))
    cv2.circle(img, tuple(average_center), 0, (0, 0, 255), 2)
    cv2.circle(img, tuple(average_center), radius, (0, 255, 0), 1)
    return img, average_center

def average_surroundings(img, col, row):
    surrounding_pixels = img[max(0, col-1):col+2, max(0, row-1):row+2]
    mask = np.all(surrounding_pixels == surrounding_pixels[0, 0], axis=-1)
    new_pixel = np.mean(surrounding_pixels[mask], axis=0)
    return tuple(new_pixel.astype(np.uint8))

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
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    return hog_image_rescaled

def find_farthest_points(image):
    # Get the coordinates of all white pixels in the image
    white_pixels = np.argwhere(np.all(image == [255, 255, 255], axis=-1))

    if len(white_pixels) < 2:
        return([0, 0], [0, 0], [0, 0])

    # Compute pairwise distances between white pixels
    dist_matrix = cdist(white_pixels, white_pixels)

    # Find the pair of white pixels with the largest distance
    max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return (white_pixels[max_dist_idx[0]], white_pixels[int(max_dist_idx[1]/2)], white_pixels[max_dist_idx[1]])


def find_circle_center(p1, p2, p3):
    if (p2[1] - p1[1]) == 0 or (p3[1] - p2[1]) == 0:
        return  np.array([0, 0])
    # Calculate the perpendicular bisectors of the sides
    mid12 = (p1 + p2) / 2
    mid23 = (p2 + p3) / 2
    slope12 = -(p2[0] - p1[0]) / (p2[1] - p1[1])
    slope23 = -(p3[0] - p2[0]) / (p3[1] - p2[1])
    intercept12 = mid12[1] - slope12 * mid12[0]
    intercept23 = mid23[1] - slope23 * mid23[0]
    x = (intercept23 - intercept12) / (slope12 - slope23)
    y = slope12 * x + intercept12
    return np.array([x, y])

def show(name,img):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 10,40)
    cv2.imshow(name,img)

def find_ceter(img_file, show_i=False):  
    img = cv2.imread(img_file)
    no_cross =  remove_cross(img)

    edges = cv2.Canny(no_cross, 40, 40)
    edges_no_cross = np.expand_dims(edges, axis=-1)
    edges_no_cross = np.repeat(edges_no_cross, 3, axis=-1)

    hog_32 = apply_hog(no_cross, pixels=(32, 32))
    hog_16 = apply_hog(no_cross, pixels=(16, 16))
    hog_8 = apply_hog(no_cross, pixels=(8, 8))

    hog_img = cv2.add(cv2.add(hog_32, hog_16, hog_8), edges_no_cross.copy())

    to_find = cv2.subtract(no_cross, hog_img)
    hough, center = detect_circles_with_hough(no_cross, to_find)
    if hough is not None:
        if show_i:
            images = np.concatenate((hog_img, hough), axis=1)
            show(img_file, images)
        return center

    hog_48 = apply_hog(no_cross, pixels=(48, 48))
    hog_24 = apply_hog(no_cross, pixels=(24, 24))
    hog_12 = apply_hog(no_cross, pixels=(12, 12))

    hog_img = cv2.add(cv2.add(hog_48, hog_24, hog_12), edges_no_cross.copy())
    to_find = cv2.subtract(no_cross, hog_img)
    hough, center = detect_circles_with_hough(no_cross, to_find)
    if hough is not None:
        if show_i:
            images = np.concatenate((hog_img, hough), axis=1)
            show(img_file, images)
        return center

    hog_64 = apply_hog(no_cross, pixels=(64, 64))

    hog_img = cv2.add( cv2.add(hog_64, hog_32),  cv2.add(hog_16, hog_8), edges_no_cross.copy())
    to_find = cv2.subtract(no_cross, hog_img)
    hough, center = detect_circles_with_hough(no_cross, to_find)
    if hough is not None:
        if show_i:
            images = np.concatenate((hog_img, hough), axis=1)
            show(img_file, images)
        return center

    hough, center = detect_circles_with_hough(no_cross, edges_no_cross.copy())
    if hough is not None:
        if show_i:
            images = np.concatenate((edges_no_cross.copy(), hough), axis=1)
            show(img_file, images)
        return center
    
    if hough is None:
                
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_dilation2 = cv2.dilate(edges_no_cross.copy(), kernel2, iterations=3)
        img_erosion2 = cv2.erode(img_dilation2, kernel, iterations=2)
        img_dilation3 = cv2.dilate(img_erosion2, kernel, iterations=100)
        img_erosion3 = cv2.erode(img_dilation3, kernel, iterations=99)

        edges = cv2.Canny(cv2.subtract(no_cross, img_erosion3), 50, 150)
        edges_with_diml = np.expand_dims(edges, axis=-1)
        edges_with_diml = np.repeat(edges_with_diml, 3, axis=-1)

        center = find_circle_center(*find_farthest_points(edges_with_diml))
        return center

if __name__ == '__main__':
    directory = 'Circle_Center_Detection'
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", action='store_true', help="show images", default=False)
    args = parser.parse_args()
    
    for subdir, dirs, files in os.walk(directory):
        for file_name in files:
            img_file = os.path.join(subdir, file_name)
            print(f"in image {file_name} center is {find_ceter(img_file, show_i=args.show)}")

        if args.show:
            cv2.waitKey()
            cv2.destroyAllWindows()