import streamlit as st
import numpy as np
import cv2

from utils import load_image
from preprocess import * 

# Title of the application
st.title('CV app with Streamlit and OpenCV')

# Sidebar settings
st.sidebar.title('Menu')
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.sidebar.title('Quantization')
# Slider for number of bits
bits = st.sidebar.slider("Number of Bits", min_value=0, max_value=8, value=4)

# Button to apply quantization
quantized_button = st.sidebar.button("Apply Quantization")

st.sidebar.title('Color Spaces')
# Dropdown for color space selection
color_space = st.sidebar.selectbox("Select Color Space", ["GRAY", "HSV", "LAB", "YUV"])

# Button to apply color space conversion
convert_button = st.sidebar.button("Apply Color Space")

st.sidebar.title('Geometric Transformation')
# Slider and button for translation
tx = st.sidebar.slider("Translation X (pixels)", min_value=-500, max_value=500, value=0)
ty = st.sidebar.slider("Translation Y (pixels)", min_value=-500, max_value=500, value=0)
translate_button = st.sidebar.button("Apply Translation")

# Slider and button for scaling
scale_x = st.sidebar.slider("Scaling Factor X", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
scale_y = st.sidebar.slider("Scaling Factor Y", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
scale_button = st.sidebar.button("Apply Scaling")

# Slider and button for rotation
angle = st.sidebar.slider("Rotation Angle (degrees)", min_value=-180, max_value=180, value=0)
rotation_button = st.sidebar.button("Apply Rotation")

# Load the template image
template_image = cv2.imread("data/examples/blank_image.jpg")

# Create two columns: Input and Output
col1, col2 = st.columns(2)

# Column headers
with col1:
    st.header("Input")

with col2:
    st.header("Output")

# Initialize images
image = None
quantized_image = None
converted_image = None
transformed_image = None

# Check if a file is uploaded
if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV image
    image = load_image(uploaded_file)

    with col1:
        # Display the uploaded image in the In column
        st.image(image, channels="BGR", caption='Original Image', use_column_width=True)

    if quantized_button and image is not None:
        # Apply uniform quantization when the button is clicked
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quantized_image = uniform_quantization(gray_image, bits)

    if convert_button and image is not None:
        # Convert the image to the selected color space
        converted_image = convert_color(image, color_space)

    # Apply translation
    if translate_button and image is not None:
        transformed_image = translate_image(image, tx, ty)
    
    # Apply scaling
    if scale_button and image is not None:
        transformed_image = scale_image(image, scale_x, scale_y)
    
    # Apply rotation
    if rotation_button and image is not None:
        transformed_image = rotate_image(image, angle)

    # Display the processed image in the Output column
    with col2:
        if converted_image is not None:
            st.image(converted_image, channels="BGR" if color_space != "GRAY" else "GRAY", caption=f'Image in {color_space} Space', use_column_width=True)
        elif quantized_image is not None:
            st.image(quantized_image, caption=f'Quantized Image ({bits} bits)', use_column_width=True)
        elif transformed_image is not None:
            st.image(transformed_image, channels="BGR" if transformed_image.ndim == 3 else "GRAY", caption='Transformed Image', use_column_width=True)
        else:
            st.image(template_image, caption='Template Image', use_column_width=True)
else:
    # Display the template image in the Input column initially
    with col1:
        st.image(template_image, caption='Blank Image', use_column_width=True)

    # Display the template image in the Output column initially
    with col2:
        st.image(template_image, caption='Blank Image', use_column_width=True)