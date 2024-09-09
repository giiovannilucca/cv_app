import cv2
import numpy as np

def uniform_quantization(image, bits):
    """
    Apply uniform quantization to the input image with the specified number of bits.
    
    Parameters:
    - image: input image (numpy array)
    - bits: number of bits for quantization (integer)
    
    Returns:
    - quantized_image: quantized image (numpy array)
    """
    levels = 2 ** bits    
    step_size = 256 // levels    
    quantized_image = (image // step_size) * step_size + step_size // 2
    
    return quantized_image

def convert_color(image, color_space):
    """
    Convert the input image to the specified color space using OpenCV.
    
    Parameters:
    - image: input image (numpy array)
    - color_space: desired color space (string)
    
    Returns:
    - converted_image: image in the specified color space (numpy array)
    """
    if color_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    else:
        raise ValueError("Unsupported color space")
    
def translate_image(image, tx, ty):
    """
    Translate the input image by (tx, ty) pixels.
    
    Parameters:
    - image: input image (numpy array)
    - tx: translation in the x direction (integer)
    - ty: translation in the y direction (integer)
    
    Returns:
    - translated_image: translated image (numpy array)
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def scale_image(image, scale_x, scale_y):
    """
    Scale the input image by scale_x and scale_y factors.
    
    Parameters:
    - image: input image (numpy array)
    - scale_x: scaling factor in the x direction (float)
    - scale_y: scaling factor in the y direction (float)
    
    Returns:
    - scaled_image: scaled image (numpy array)
    """
    rows, cols = image.shape[:2]
    M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    scaled_image = cv2.warpAffine(image, M, (int(cols * scale_x), int(rows * scale_y)))
    return scaled_image

def rotate_image(image, angle, center=None, scale=1.0):
    """
    Rotate the input image by a specified angle.
    
    Parameters:
    - image: input image (numpy array)
    - angle: angle to rotate (degrees)
    - center: center of rotation (tuple of floats), default is the center of the image
    - scale: scaling factor, default is 1.0
    
    Returns:
    - rotated_image: rotated image (numpy array)
    """
    rows, cols = image.shape[:2]
    if center is None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def histogram_equalization_gray(image):
    """
    Apply histogram equalization to a grayscale image.
    
    Parameters:
    - image: input grayscale image (numpy array)
    
    Returns:
    - equalized_image: image after histogram equalization (numpy array)
    """
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def histogram_equalization_color(image):
    """
    Apply histogram equalization to a color image.
    
    Parameters:
    - image: input color image (numpy array)
    
    Returns:
    - equalized_image: color image after histogram equalization (numpy array)
    """
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_image

def apply_filter(image, filter_type, kernel_size):
    """
    Apply the specified filter to the input image using the given kernel size.
    
    Parameters:
    - image: input image (numpy array)
    - filter_type: type of filter to apply (string)
    - kernel_size: size of the kernel (integer)
    
    Returns:
    - filtered_image: filtered image (numpy array)
    """
    if filter_type == 'Mean':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'Median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'Gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'Bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError("Unsupported filter type")