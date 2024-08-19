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
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    return img