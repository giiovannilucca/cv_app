import cv2
import numpy as np
import streamlit as st

from utils import *
from preprocess import * 
from edge_detection import *

# Title of the application
st.title('CV app with Streamlit and OpenCV')

# Sidebar settings
st.sidebar.title('Menu')
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.sidebar.title('Quantization')
bits = st.sidebar.slider("Number of Bits", min_value=0, max_value=8, value=4)
quantized_button = st.sidebar.button("Apply Quantization")

st.sidebar.title('Color Spaces')
color_space = st.sidebar.selectbox("Select Color Space", ["GRAY", "HSV", "LAB", "YUV"])
convert_button = st.sidebar.button("Apply Color Space")

st.sidebar.title('Geometric Transformation')
tx = st.sidebar.slider("Translation X (pixels)", min_value=-500, max_value=500, value=0)
ty = st.sidebar.slider("Translation Y (pixels)", min_value=-500, max_value=500, value=0)
translate_button = st.sidebar.button("Apply Translation")

scale_x = st.sidebar.slider("Scaling Factor X", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
scale_y = st.sidebar.slider("Scaling Factor Y", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
scale_button = st.sidebar.button("Apply Scaling")

angle = st.sidebar.slider("Rotation Angle (degrees)", min_value=-180, max_value=180, value=0)
rotation_button = st.sidebar.button("Apply Rotation")

st.sidebar.title('Enhancement')
enhancement_method = st.sidebar.selectbox(
    "Select Enhancement Method",
    ["Histogram Equalization (Grayscale)", "Histogram Equalization (Color)"]
)
enhance_button = st.sidebar.button("Apply Enhancement")

st.sidebar.title('Filters')
kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=5, step=2)
filter_type = st.sidebar.selectbox("Select Filter", ["Mean", "Median", "Gaussian", "Bilateral"])
filter_button = st.sidebar.button("Apply Filter")

st.sidebar.title('Edge Detection')
edge_method = st.sidebar.selectbox("Select Edge Detection Method", ["Sobel", "Laplacian", "Canny"])
threshold1 = None
threshold2 = None
if edge_method == "Canny":
    threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
    threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
apply_edge_button = st.sidebar.button("Apply Edge Detection")

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
image = quantized_image = converted_image = None
transformed_image = enhanced_image = filtered_image = None
edge_image = None

# Check if a file is uploaded
if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV image
    image = load_image(uploaded_file)
    image_type = detect_image_type(image)

    with col1:
        # Display the uploaded image in the In column
        st.image(image, channels="BGR" if image_type == 'color' else 'GRAY', caption='Original Image', use_column_width=True)

    if quantized_button and image is not None:
        # Apply uniform quantization when the button is clicked
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quantized_image = uniform_quantization(gray_image, bits)

    if convert_button and image is not None:
        # Convert the image to the selected color space
        converted_image = convert_color(image, color_space)

    if translate_button and image is not None:
        transformed_image = translate_image(image, tx, ty)
    
    if scale_button and image is not None:
        transformed_image = scale_image(image, scale_x, scale_y)
    
    if rotation_button and image is not None:
        transformed_image = rotate_image(image, angle)

    if enhance_button and image is not None:
        if enhancement_method == "Histogram Equalization (Grayscale)":
            if len(image.shape) == 2:  # Grayscale check
                enhanced_image = histogram_equalization_gray(image)
            else:
                st.warning("Please upload a grayscale image for this enhancement.")
        elif enhancement_method == "Histogram Equalization (Color)":
            if len(image.shape) == 3:  # Color image check
                enhanced_image = histogram_equalization_color(image)
            else:
                st.warning("Please upload a color image for this enhancement.")        

    if filter_button and image is not None:
        filtered_image = apply_filter(image, filter_type, kernel_size)

    if apply_edge_button and image is not None:
        if edge_method == "Sobel":
            edge_image = sobel_edge_detection(image)
        elif edge_method == "Laplacian":
            edge_image = laplacian_edge_detection(image)
        elif edge_method == "Canny" and threshold1 is not None and threshold2 is not None:
            edge_image = canny_edge_detection(image, threshold1, threshold2)

    # Display the processed image in the Output column
    with col2:
        if converted_image is not None:
            st.image(converted_image, channels="BGR" if color_space != "GRAY" else "GRAY", caption=f'Image in {color_space} Space', use_column_width=True)
        elif quantized_image is not None:
            st.image(quantized_image, caption=f'Quantized Image ({bits} bits)', use_column_width=True)
        elif transformed_image is not None:
            st.image(transformed_image, channels="BGR" if transformed_image.ndim == 3 else "GRAY", caption='Transformed Image', use_column_width=True)
        elif enhanced_image is not None:
            st.image(enhanced_image, channels="BGR" if image_type == 'color' else 'GRAY', caption=f'Enhanced Image ({enhancement_method})', use_column_width=True)
        elif filtered_image is not None:
            st.image(filtered_image, channels="BGR" if image_type == 'color' else 'GRAY', caption=f'{filter_type} Filter Applied (Kernel Size: {kernel_size})', use_column_width=True)
        elif edge_image is not None:
            st.image(edge_image, caption=f'{edge_method} Edge Detection', use_column_width=True)
        else:
            st.image(template_image, caption='Template Image', use_column_width=True)
else:
    # Display the template image in the Input column initially
    with col1:
        st.image(template_image, caption='Blank Image', use_column_width=True)

    # Display the template image in the Output column initially
    with col2:
        st.image(template_image, caption='Blank Image', use_column_width=True)