import cv2
import numpy as np

from utils import is_color_image

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to the input image.

    Parameters:
    - image: input image (numpy array)

    Returns:
    - sobel_edges: edges detected using Sobel (numpy array)
    """
    if is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_edges = cv2.magnitude(sobelx, sobely)
    
    sobel_edges = np.uint8(np.absolute(sobel_edges))
    
    return sobel_edges

def laplacian_edge_detection(image):
    """
    Apply Laplacian edge detection to the input image.

    Parameters:
    - image: input image (numpy array)

    Returns:
    - laplacian_edges: edges detected using Laplacian (numpy array)
    """
    if is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)
    
    laplacian_edges = np.uint8(np.absolute(laplacian_edges))
    
    return laplacian_edges

def canny_edge_detection(image, threshold1, threshold2):
    """
    Apply Canny edge detection to the input image.

    Parameters:
    - image: input image (numpy array)
    - threshold1: first threshold for hysteresis procedure (integer)
    - threshold2: second threshold for hysteresis procedure (integer)

    Returns:
    - canny_edges: edges detected using Canny (numpy array)
    """
    if is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    canny_edges = cv2.Canny(image, threshold1, threshold2)
    
    return canny_edges
