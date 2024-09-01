import numpy as np
import cv2

def load_image(image_file):
    """
    Load an image from an uploaded file and decode it into an OpenCV image.

    Parameters:
    - image_file: the uploaded image file (Streamlit file uploader object)

    Returns:
    - img: the decoded image (numpy array)
    """
    # Read the uploaded file as bytes
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    
    # Decode the image using OpenCV
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    return img

def detect_image_type(image):
    """
    Detect whether the input image is color or grayscale.
    
    Parameters:
    - image: input image (numpy array)
    
    Returns:
    - image_type: 'color' if the image is color, 'grayscale' otherwise
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return 'color'
    elif len(image.shape) == 2:
        return 'grayscale'
    else:
        raise ValueError("Unsupported image format.")