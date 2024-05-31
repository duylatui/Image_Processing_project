import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Điều chỉnh cân bằng màu sắc RGB
def adjust_rgb_color_balance(image, red_balance, green_balance, blue_balance):
    # Lưu ý: 'image' ở đây là một array của numpy
    adjusted_img = image.copy()
    adjusted_img[:,:,0] = np.clip(image[:,:,0] * blue_balance, 0, 255)
    adjusted_img[:,:,1] = np.clip(image[:,:,1] * green_balance, 0, 255)
    adjusted_img[:,:,2] = np.clip(image[:,:,2] * red_balance, 0, 255)
    return adjusted_img

# Điều chỉnh gamma
def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Hàm chính để chạy ứng dụng Streamlit
def main():
    st.title("Color Balance")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Tạo sliders để điều chỉnh màu sắc và gamma
        red_balance = st.slider('Red Balance', min_value=0.0, max_value=2.0, value=1.0)
        green_balance = st.slider('Green Balance', min_value=0.0, max_value=2.0, value=1.0)
        blue_balance = st.slider('Blue Balance', min_value=0.0, max_value=2.0, value=1.0)
        gamma = st.slider('Gamma', min_value=0.1, max_value=3.0, value=1.0)

        # Áp dụng điều chỉnh
        adjusted_image = adjust_rgb_color_balance(image, red_balance, green_balance, blue_balance)
        adjusted_image = adjust_gamma(adjusted_image, gamma)

        # Hiển thị ảnh
        st.image(adjusted_image, channels="RGB", use_column_width=True)

        # Lưu ảnh đã chỉnh sửa
        if st.button('Save Image'):
            im = Image.fromarray(adjusted_image)
            im.save('adjusted_image.jpg')
            st.success('Saved Image as adjusted_image.jpg')

if __name__ == "__main__":
    main()
