import streamlit as st
import cv2
import numpy as np
# from matplotlib import pyplot as plt

# Function to display images using Streamlit
def st_display_image(image, caption=''):
    st.image(image, caption=caption, use_column_width=True)

# Function to read an image from the uploaded file
def read_uploaded_image(uploaded_image):
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    return image

# Main Streamlit app
def main():
    st.title('Image Segmentation with Streamlit')

    st.sidebar.header('Upload an Image')
    uploaded_image = st.sidebar.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        original_image = read_uploaded_image(uploaded_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        # st_display_image(original_image, caption='Original Image')

        st.sidebar.header('Image Segmentation Options')
        image_processing_option = st.sidebar.selectbox('Choose an image processing technique:',
                                                        ['Canny Edge Detection', 'Thresholding', 'GrabCut', 'Watershed'])

        st.sidebar.header('Original Image')
        st.sidebar.image(original_image)
        
        if image_processing_option == 'Canny Edge Detection':
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1=30, threshold2=100)
            st.header('Edge Segmentation')
            st_display_image(edges, caption='Edge Segmentation')

        elif image_processing_option == 'Thresholding':
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            st.subheader('Thresholding Image using Otsu thresholding')
            st_display_image(bin_img, caption='Grayscale Image')

            th, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            st.subheader('Thresholding Image using Binary thresholding')
            st_display_image(dst, caption='Binary thresholding')
            
            th, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            st.subheader('Thresholding Image using Binary Inverse thresholding')
            st_display_image(dst, caption='Binary Inverse thresholding')

            th, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
            st.subheader('Thresholding Image using THRESH_TRUNC')
            st_display_image(dst, caption='THRESH_TRUNC')
            
            th, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
            st.subheader('Thresholding Image using THRESH_TOZERO')
            st_display_image(dst, caption='THRESH_TOZERO')
            
            th, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
            st.subheader('Thresholding Image using THRESH_TOZERO_INV')
            st_display_image(dst, caption='THRESH_TOZERO_INV')

            

        elif image_processing_option == 'GrabCut':
            mask = np.zeros(original_image.shape[:2], np.uint8)
            backgroundModel = np.zeros((1, 65), np.float64)
            foregroundModel = np.zeros((1, 65), np.float64)
            rectangle = (20, 100, 250, 200)

            cv2.grabCut(original_image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result_image = original_image * mask2[:, :, np.newaxis]

            st.header('Image after applying GrabCut Algorithm')
            st_display_image(result_image, caption='Image after GrabCut')

        elif image_processing_option == 'Watershed':
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

            dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
            sure_fg = sure_fg.astype(np.uint8)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers += 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(original_image, markers)

            labels = np.unique(markers)
            coins = []

            for label in labels[2:]:
                target = np.where(markers == label, 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                coins.append(contours[0])

            img = cv2.drawContours(original_image, coins, -1, color=(0, 23, 223), thickness=2)
            st.header('Watershed Segmented Output Image')
            st_display_image(img, caption='Watershed Segmented Image')

if __name__ == '__main__':
    main()