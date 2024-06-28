import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import filters
import histograms
import frequency

st. set_page_config(layout="wide")


def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Image Filtering & Edge Detection
        </h1>
    """, unsafe_allow_html=True
                )

    st.caption("""
        <p style='text-align: center'>
        by team 18
        </p>
    """, unsafe_allow_html=True
               )


def main():

    selected = option_menu(
        menu_title=None,
        options=['Filtering', 'Histograms',
                 'Mixer', 'Thresholding', 'Frequency'],
        orientation="horizontal"
    )
    if selected == "Filtering":

        with st.sidebar:
            image = st.file_uploader("Upload Image", type=[
                                     'jpg', 'jpeg', 'png'])

            # create noise form to choose noise to add to the image
            noise_form = st.form("noise form")
            with noise_form:
                noise = st.radio("Choose Noise", options=[
                                 'Uniform', 'Gaussian', 'Salt & Pepper'])
                add_noise = st.form_submit_button("Add")

            # create filter form to choose filter to apply to the image
            noise_filter_form = st.form("noise filter form")
            with noise_filter_form:
                noise_filter = st.radio("Choose Noise Filter", options=[
                    'Average', 'Gaussian', 'Median'])
                filter_kernel = st.slider("Kernel Size", 3, 15, 3, 2)
                apply_noise_filter = st.form_submit_button("Apply")

            # create edge form to choose edge detection filter to apply to the image
            edge_filter_form = st.form("edge filter form")
            with edge_filter_form:
                edge_filter = st.radio("Choose Edge Detection Filter", options=[
                    'Sobel', 'Roberts', 'Prewitt', 'Canny'])
                apply_edge_filter = st.form_submit_button("Apply")
        img_col, edited_col = st.columns(2)
        if image:
            with img_col:
                st.image(image, use_column_width=True)

        if noise == "Gaussian":
            with noise_form:
                sigma = st.slider("Sigma", 0.0, 2.0, 0.5, 0.1)
            if add_noise:
                img = filters.read_image(f"Images/{image.name}")
                noisy_img = filters.gaussian_noise(img, sigma)
                filters.write_image(noisy_img, "noise", noise)
                with edited_col:
                    st.image(
                        f"Images/img with {noise} noise.jpg", use_column_width=True)

        elif noise == "Salt & Pepper":

            if add_noise:
                img = filters.read_image_grayscale(f'Images/{image.name}')
                noisy_img = filters.salt_n_pepper_noise(img)
                filters.write_image(noisy_img, "noise", noise)
                with edited_col:
                    st.image(
                        f"Images/img with {noise} noise.jpg", use_column_width=True)

        elif noise == "Uniform":
            with noise_form:
                max = st.slider("Max", 1, 255, 100, 1)
            if add_noise:
                img = filters.read_image(f'Images/{image.name}')
                noisy_img = filters.uniform_noise(img, max)
                filters.write_image(noisy_img, "noise", noise)
                with edited_col:
                    st.image(
                        f"Images/img with {noise} noise.jpg", use_column_width=True)

        if noise_filter == "Average" and apply_noise_filter:
            img = filters.read_image(f'Images/{image.name}')
            filtered_img = filters.average_filter(img, filter_kernel)
            filters.write_image(filtered_img, "filter", noise_filter)
            with edited_col:
                st.image(
                    f"Images/img with {noise_filter} filter.jpg", use_column_width=True)

        elif noise_filter == "Median" and apply_noise_filter:
            img = filters.read_image_grayscale(f'Images/{image.name}')
            filtered_img = filters.median_filter(img, filter_kernel)
            filters.write_image(filtered_img, "filter", noise_filter)
            with edited_col:
                st.image(
                    f"Images/img with {noise_filter} filter.jpg", use_column_width=True)

        elif noise_filter == "Gaussian" and apply_noise_filter:
            img = filters.read_image(f'Images/{image.name}')
            filtered_img = filters.gaussian_filter(img, filter_kernel)
            filters.write_image(filtered_img, "filter", noise_filter)
            with edited_col:
                st.image(
                    f"Images/img with {noise_filter} filter.jpg", use_column_width=True)

        if edge_filter == "Sobel" and apply_edge_filter:
            img = filters.read_image_for_edge(f'Images/{image.name}')
            filtered_img, _ = filters.sobel_filter(img)
            filters.write_image(filtered_img, "edge", edge_filter)
            with edited_col:
                st.image(
                    f"Images/img with {edge_filter} edge.jpg", use_column_width=True)

        elif edge_filter == "Canny" and apply_edge_filter:
            img = filters.read_image_for_edge(f'Images/{image.name}')
            gradientMat, thetaMat = filters.sobel_filter(img)
            filtered_img = filters.canny_filter(gradientMat, thetaMat)
            filters.write_image(filtered_img, "edge", edge_filter)
            with edited_col:
                st.image(
                    f"Images/img with {edge_filter} edge.jpg", use_column_width=True)

        elif edge_filter == "Roberts" and apply_edge_filter:
            img = filters.read_image_for_edge(f'Images/{image.name}')
            filtered_img = filters.robert_filter(img)
            filters.write_image(filtered_img, "edge", edge_filter)
            with edited_col:
                st.image(
                    f"Images/img with {edge_filter} edge.jpg", use_column_width=True)

        elif edge_filter == "Prewitt" and apply_edge_filter:
            img = filters.read_image_for_edge(f'Images/{image.name}')
            filtered_img = filters.prewitt_filter(img)
            filters.write_image(filtered_img, "edge", edge_filter)
            with edited_col:
                st.image(
                    f"Images/img with {edge_filter} edge.jpg", use_column_width=True)

    elif selected == "Histograms":
        img_col, edited_col = st.columns(2)

        with st.sidebar:
            image = st.file_uploader("Upload Image", type=[
                                     'jpg', 'jpeg', 'png'])
            selected_data = st.selectbox("Select a histogram to display",
                                         ["Gray Scale", "Red Channel", "Green Channel", "Blue Channel", "All Histograms"])

            select_mode = st.selectbox("Select a mode for equalization & normalization",
                                       ["Gray Scale", "RGB"])

            normalization_form = st.form("Normalization Form")
            with normalization_form:
                normalize = st.form_submit_button("Normalize")
            equalization_form = st.form("Equalization Form")
            with equalization_form:
                equalize = st.form_submit_button("Equalize")
            to_gray_form = st.form("To Gray Form")
            with to_gray_form:
                to_gray = st.form_submit_button("Change image into Grayscale")
        if image:
            with img_col:
                st.image(image, use_column_width=True)
                img = filters.read_image(f"Images/{image.name}")
            with edited_col:
                if normalize:
                    if (select_mode == "Gray Scale"):
                        normalized_img = histograms.gray_normalize(img)
                    elif (select_mode == "RGB"):
                        normalized_img = histograms.normalize(img)
                    st.image([normalized_img], caption=[
                             'Normalized Image'], use_column_width=True)
                if equalize:
                    if (select_mode == "Gray Scale"):
                        equalized_img = histograms.gray_equalize(img)
                    elif (select_mode == "RGB"):
                        equalized_img = histograms.rgb_equalize(img)
                    st.image([equalized_img], caption=[
                             'Equalized Image'], use_column_width=True)
                if to_gray:
                    #gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray2 = frequency.convertcolortograyscale(img)[0]
                    # print(type(gray_img[0,2]))
                    # print(type(gray2[0,2]))
                    #st.image([gray_img], caption=['Grayscaled Image'],use_column_width=True)
                    st.image([gray2], caption=['Grayscaled Image'],
                             use_column_width=True)
                    # print(gray_img,"Auto")
                    # print(gray2,"implemented")

            hist_col, curve_col = st.columns(2)
            gray_hist = histograms.gray_hist(img)
            hist_blue, hist_green, hist_red = histograms.rgb_hist(img)
            hist_fig = histograms.plot_hist(
                gray_hist, hist_red, hist_green, hist_blue)
            curve_fig = histograms.plot_curve(
                gray_hist, hist_red, hist_green, hist_blue)
            with hist_col:
                histograms.select_data(selected_data, hist_fig)
            with curve_col:
                histograms.select_data(selected_data, curve_fig)

    elif selected == "Mixer":
        with st.sidebar:

            image1 = st.file_uploader("Upload First Image", type=[
                'jpg', 'jpeg', 'png'])

            image2 = st.file_uploader("Upload Second Image", type=[
                'jpg', 'jpeg', 'png'])
            threshold0 = st.slider("threshold", 0, 100, 20, 10)
            mixer_form = st.form("mixer form")
            

        image1_col, image2_col = st.columns(2)
        if image1:
            with image1_col:
                st.image(image1, use_column_width=True)

        if image2:
            with image2_col:
                st.image(image2, use_column_width=True)

        if image1 and image2:
            
            hybrid_img = frequency.hybrid(
                f'Images/{image1.name}', f'Images/{image2.name}',threshold0)
            filters.write_image(hybrid_img, "hib", 'hybrid')
            st.image(
                f"Images/img with hybrid hib.jpg", use_column_width=True)

    elif selected == "Thresholding":
        img_col, edited_col = st.columns(2)

        with st.sidebar:
            image = st.file_uploader("Upload Image", type=[
                                     'jpg', 'jpeg', 'png'])
            select_mode = st.selectbox("Select thresholding Mode",
                                       ["Global", "Local"])
            if select_mode == "Global":
                threshold = st.slider("threshold", 0, 255, 127, 1)
            else:
                blocksize = st.slider("blocksize", 1, 100, 15, 1)
            Thresholdingform = st.form("thresholding Form")
            with Thresholdingform:
                thresholdbutton = st.form_submit_button("Apply")
        if image:
            with img_col:
                st.image(image, caption=[
                         'Original Image'], use_column_width=True)
                img = filters.read_image_grayscale(f"Images/{image.name}")
            with edited_col:
                if thresholdbutton:
                    if (select_mode == "Global"):
                        threshold_img = frequency.globalthresholding(
                            img, int(threshold))
                    elif (select_mode == "Local"):
                        threshold_img = frequency.localthresholding(
                            img, int(blocksize))
                    st.image([threshold_img], caption=[
                             'Thresholded Image'], use_column_width=True)

    elif selected == "Frequency":
        img_col, edited_col = st.columns(2)
        with st.sidebar:
            image = st.file_uploader("Upload Image", type=[
                                     'jpg', 'jpeg', 'png'])
            select_mode = st.selectbox("Select filter",
                                       ["Highpass", "Lowpass"])
            threshold2 = st.slider("threshold", 10, 100, 50, 10)

            # freqform = st.form("freq Form")
            # with freqform:
            #     filterbutton = st.form_submit_button("Apply")
        if image:
            with img_col:
                st.image(image, caption=[
                         'Original Image'], use_column_width=True)
                img = filters.read_image_grayscale(f"Images/{image.name}")
            with edited_col:
                # if filterbutton:
                    if (select_mode == "Lowpass"):
                        Fshift, img = frequency.fourier(f'Images/{image.name}')
                        Gshiftl, s = frequency.lowpass(
                            Fshift, img, int(threshold2))
                        edited_img = frequency.inversefourier(Gshiftl)
                        filters.write_image(edited_img, "freq", select_mode)

                    elif (select_mode == "Highpass"):
                        Fshift, img = frequency.fourier(f'Images/{image.name}')
                        q, H = frequency.lowpass(Fshift, img, int(threshold2))
                        Gshift = frequency.highpass(Fshift, H)
                        edited_img = frequency.inversefourier(Gshift)
                        filters.write_image(edited_img, "freq", select_mode)

                    
                    st.image(
                        f"Images/img with {select_mode} freq.jpg", use_column_width=True)


if __name__ == '__main__':
    main()
