import streamlit as st
import numpy as np
from skimage.color import gray2rgb
import skimage.draw
from activeContour import active_contour, init_square_snake, init_circle_snake
from streamlit_option_menu import option_menu
import hough_transform as ht
import cv2

st.set_page_config(layout="wide")

def main():
    selected = option_menu(
        menu_title=None,
        options=['Hough Transform', 'Active Contour'],
        orientation="horizontal"
    )

    if selected == "Hough Transform":
        with st.sidebar:
            img = st.file_uploader("Upload Image", type=[
                                     'jpg', 'jpeg', 'png'])
            form = st.form("hough")
            with form:
                option = st.selectbox("Select Hough Transform", [
                                  "Line", "Circle", "Ellipse"])
                apply = st.form_submit_button("Apply")
            
        image_col, edited_col = st.columns(2)
        
        if img:
            with image_col:
                st.image(img,use_column_width=True)
            
        if apply and option == "Line":
            editied_image = cv2.imread(f"Images/{img.name}",cv2.IMREAD_GRAYSCALE)
            canny_img = ht.Canny(editied_image)
            accumulator, thetas, rhos = ht.houghLine(canny_img)
            ht.add_lines(editied_image,accumulator,thetas,rhos)
            with edited_col:
                st.image("result.jpg",use_column_width=True)
            
        elif apply and option == "Circle":
            pass
        elif apply and option == "Ellipse":
            pass

    elif selected == "Active Contour":
        st.title("Active Contour")

        # Create a blank image   #it can be a blank image or the user can choose another image
        image = np.zeros((1000, 1000))
        image[50:150, 50:150] = 1
        image[60:140, 60:140] = 0

        # example
        # image = skimage.io.imread("C:/Users/Alaa Yasser/OneDrive/Desktop/computer vision/assigment2 py/a02-team_18/Images/example.jpg")

        # # # Resize the image to 1000x1000
        # image = resize(image, (6000, 6000))
        # # print(image.shape)

        # # Define the center and radius of the circle
        # circle_center = (2500, 2500)
        # circle_radius =2000

        # img_gray = rgb2gray(image)

        # # Initialize a circle snake with 50 points
        # snake = init_circle_snake(circle_center, circle_radius, 500)

        # # Use the snake as the initial contour for your active contour algorithm
        # snake_after = active_contour(img_gray, snake,gamma=5 , max_num_iter=100)

        # The can define initial points for the snake or a circle or a square
        # example # Define initial snake
        # snake = np.array([[100, 100], [100, 110], [100, 120], [100, 130], [100, 140], [100, 150], [100, 160],
        #                   [110, 160], [120, 160], [130, 160], [140, 160], [150, 160], [160, 160], [160, 150],
        #                   [160, 140], [160, 130], [160, 120], [160, 110], [160, 100], [150, 100], [140, 100],
        #                   [130, 100], [120, 100], [110, 100]])
        # # Initialize a circle snake with 50 points
        snake = init_square_snake((100, 100), 100)
        # Run active contour algorithm
        snake = active_contour(image, snake, gamma=-5,  max_num_iter=1)

        # Convert the grayscale image to RGB
        img_rgb = gray2rgb(image)

        # Draw the contour on the image
        for i in range(len(snake)-1):
            rr, cc = skimage.draw.line(
                *snake[i].astype(int), *snake[i+1].astype(int))
            # rr, cc = skimage.draw.line(snake[i, 0], snake[i, 1], snake[i+1, 0], snake[i+1, 1])
            img_rgb[rr, cc] = [1, 0, 0]  # Red color for the contour

        # Display the image and contour using Streamlit
        st.image(img_rgb, caption="Image with Active Contour",
                 use_column_width=True)


if __name__ == '__main__':
    main()
