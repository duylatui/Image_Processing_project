import cv2
import numpy as np

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
# Streamlit UI
st.title("Chroma Keying Application")


def process_image(uploaded_image, background, color_choice): 
      # Đảm bảo cùng kích thước
    if background.shape[:2] != uploaded_image.shape[:2]:
        background = cv2.resize(background, (uploaded_image.shape[1], uploaded_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Chuyển ảnh tải lên sang HSV
    hsv = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2HSV)
    # Đọc và xử lý ảnh tại đâydef process_image(uploaded_image, background, color_choice): pass
    # Trả về ảnh đã tách 
    # Xác định khoảng màu dựa trên color_choice
    if color_choice == "Green":
        lower_color = np.array([40, 40, 40])
        upper_color = np.array([70, 255, 255])
    elif color_choice == "Blue":
        lower_color = np.array([100, 150, 0])
        upper_color = np.array([140, 255, 255])
    elif color_choice == "Red":
        # Màu đỏ có thể phức tạp hơn vì nó nằm ở hai đầu của phổ HSV
        lower_color = np.array([0, 120, 70])
        upper_color = np.array([10, 255, 255])
        lower_color2 = np.array([170, 120, 70])
        upper_color2 = np.array([180, 255, 255])
    elif color_choice == "White":
        # Khoảng màu cho màu trắng
        # Độ sáng cao và độ bão hòa thấp
        lower_color = np.array([0, 0, 235])
        #upper_color = np.array([180, 55, 255])
        #bỏ bớt phạm vi sáng và giảm độ bão hòa
        upper_color = np.array([180, 20, 255])
    
    # Tạo mặt nạ cho nền
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask_inv = cv2.bitwise_not(mask)

    # Trích xuất đối tượng từ ảnh tải lên
    object_extracted = cv2.bitwise_and(uploaded_image, uploaded_image, mask=mask_inv)

    # Kết hợp đối tượng với ảnh nền
    background_masked = cv2.bitwise_and(background, background, mask=mask)
    result = cv2.add(background_masked, object_extracted)

    return object_extracted


def get_background_path(color):
    # Xác định thư mục hiện tại của tệp script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Xây dựng đường dẫn tương đối đến thư mục BackgroundPlain
    background_dir = os.path.join(current_dir, '../pages/BackgroundPlain/')
    # Xây dựng đường dẫn đầy đủ đến tệp ảnh nền dựa trên màu được chọn
    background_path = os.path.join(background_dir, f'{color.lower()}.jpg')
    return background_path


uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
color_choice = st.selectbox("Select the background color to remove", ["Green", "Blue", "Red", "White"])


    
if uploaded_image is not None and color_choice:
    # Chuyển ảnh tải lên thành định dạng có thể xử lý
    image_stream = io.BytesIO(uploaded_image.getvalue())
    #image = Image.open(image_stream).convert("RGB")


    image1 = Image.open(image_stream).convert("RGB")
    image = np.array(image1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #image = cv2.imread(uploaded_image)
    
    #print(color_choice)
    background_path = get_background_path(color_choice)
    background = cv2.imread(background_path)
    # Gọi hàm xử lý ảnh
    processed_image = process_image(image, background,color_choice)
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    # Hiển thị ảnh
    st.image(processed_image_rgb, use_column_width=True)


