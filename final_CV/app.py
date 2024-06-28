import streamlit as st 
from streamlit_option_menu import option_menu
import pickle
import functions
from PIL import Image
import cv2
import numpy as np


with open("model.pkl","rb") as f:
        model = pickle.load(f)   

st.set_page_config(layout="wide")

def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Image Recognition App
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
        options=['Upload Photo', 'Camera Input', "ROC Curves"],
        orientation="horizontal"
    )
    
    if selected == "Upload Photo":
        image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"]) 
        col1, col2 = st.columns(2)
        if image:
            with col1:
                st.image(image)
            pred = functions.predictor(f"test/{image.name}",functions.mean_face, functions.eigvecs, model)
            with col2:
                st.success(pred)
                
    
    elif selected == "Camera Input":
        col1, col2 = st.columns(2)
        with col1:
            img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f"test/{img_file_buffer.name}",cv2_img)
            pred = functions.predictor(f"test/{img_file_buffer.name}",functions.mean_face, functions.eigvecs, model)
            print(pred)
            with col2:
                st.success(pred)
    elif selected == "ROC Curves":
        image = Image.open("images\output.png")
        st.image(image, caption="ROC curve", width=700)
        

    
    
        
if __name__ == '__main__':
    # head()
    main()
        