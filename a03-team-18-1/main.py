import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from Harrisandlambda import *
from SIFT import *
import FeatureMatching as f


st.set_page_config(layout="wide")


def main():
    selected = option_menu(
        menu_title=None,
        options=['Harris \ Lambda', 'SIFT', 'Feature Matching'],
        orientation="horizontal"
    )

    if selected == "Harris \ Lambda":
        with st.sidebar:

            img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            Operator = st.radio(
                        "choose your operator",
                        ('Harris', 'Lambda-'))
            if (Operator == 'Harris'):
                    K = st.number_input("Enter K", 0.01, 0.25, 0.05, 0.01)

            harris_form = st.form("harris")
            with harris_form:

                window_size = st.slider("Choose window size", 3, 9, 3, 2)
                Q = st.number_input(
                     "Higher precision step",
                        min_value=.97,
                        max_value=.9999,
                        step=.0001,
                        value=0.999,
                        format="%.4f")
                apply = st.form_submit_button("Apply")

        image_col, edited_col = st.columns(2)

        if img:
            with image_col:
                st.image(img, use_column_width=True)

        if apply:
            if (Operator == 'Harris'):
                print("entered harris")
                src = cv2.imread(f"Images/{img.name}",1)
                edited= harrisoperator(src,k=K,window=window_size,q=Q)

            elif (Operator == 'Lambda-'):
                print("entered lambda")
                src = cv2.imread(f"Images/{img.name}",1)
                edited= lambdamin(src,window=window_size,q=Q)

              

            with edited_col:
                st.image(f"images/{edited}",1, use_column_width=True)

    elif selected == "SIFT":
        with st.sidebar:
            img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        image_col, edited_col = st.columns(2)

        if img:
            with image_col:
                st.image(img, use_column_width=True)
            with edited_col:
                src = cv2.imread(f"Images/{img.name}", cv2.IMREAD_COLOR)
                t1 = time.time()
                keypoints, descriptors = SIFT.generateFeatures(src)
                t2 = time.time()
                print("Execution time of SIFT is {} sec".format(t2 - t1))
                rgbImg = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

                fig, ax = plt.subplots()
                ax.imshow(rgbImg)
                for pnt in keypoints:
                    ax.scatter(pnt.pt[0], pnt.pt[1], s=pnt.size, c="red")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
    elif selected == "Feature Matching":
        with st.sidebar:
            img_1 = st.file_uploader("Upload Image", type=[
                                     "jpg", "jpeg", "png"], key=1)
            img_2 = st.file_uploader("Upload Image", type=[
                                     "jpg", "jpeg", "png"], key=2)
            method = st.selectbox("Method",['SSD','NCC'])
        image1_col, image2_col = st.columns(2)

        if img_1 and img_2:
            with image1_col:
                st.image(img_1, use_column_width=True)
            with image2_col:
                st.image(img_2, use_column_width=True)

            first_image = cv2.imread(f"Images/{img_1.name}")
            second_image = cv2.imread(f"Images/{img_2.name}")

            first_image = cv2.resize(first_image, (256, 256))
            second_image = cv2.resize(second_image, (256, 256))

            first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
            second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            kp1, descriptor1 = sift.detectAndCompute(first_image, None)
            kp2, descriptor2 = sift.detectAndCompute(second_image, None)
            t1 = time.time()
            matched_features = f.feature_matching_temp(
                descriptor1, descriptor2, method)
            t2 = time.time()
            print("Execution time of Feature Matching is {} sec".format(t2 - t1))
            matched_features = sorted(
                matched_features, key=lambda x: x.distance, reverse=True)
            matched_image = cv2.drawMatches(
                first_image, kp1, second_image, kp2, matched_features[:30], second_image, flags=2)

            st.image(matched_image, use_column_width=True)


if __name__ == '__main__':
    main()
