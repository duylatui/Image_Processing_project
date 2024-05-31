import streamlit as st
st.set_page_config(
    page_title="Do An Cuoi Ky",
    page_icon="",
)
image_url = "https://fit.hcmute.edu.vn/Resources/Images/SubDomain/fit/HCMUTE-fit.png" 
st.image(image_url, use_column_width=True)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>ĐỒ ÁN CUỐI KÌ</h1>
    </div>
    <div style="text-align: center;">
        <h1>BỘ MÔN: XỬ LÝ ẢNH SỐ</h1>
    </div>
        <div style="text-align: center;">
        <h3>GVHD: Ths. Trần Tiến Đức</h1>
    </div>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
        <div style="border: 2px solid #000000; padding: 10px; border-radius: 10px;">
            <div style="display: grid; grid-template-columns: auto auto;">
                <p style="margin-bottom: 10px;">Họ và tên:</p>
                <p style="margin-bottom: 10px;">Đào Lê Huy Giáp</p>
            </div>
            <div style="display: grid; grid-template-columns: auto auto;">
                <p style="margin-bottom: 10px;">MSSV:</p>
                <p style="margin-bottom: 10px;">21133028</p>
            </div>
        </div>
        <div style="border: 2px solid #000000; padding: 10px; border-radius: 10px;">
            <div style="display: grid; grid-template-columns: auto auto;">
                <p style="margin-bottom: 10px;">Họ và tên:</p>
                <p style="margin-bottom: 10px;">Nguyễn Đức Duy</p>
            </div>
            <div style="display: grid; grid-template-columns: auto auto;">
                <p style="margin-bottom: 10px;">MSSV:</p>
                <p style="margin-bottom: 10px;">21133020</p>
            </div>
        </div>
    </div>
    <hr style="margin-top: 20px; margin-bottom: 20px;">
    """, unsafe_allow_html=True
)
st.markdown(
    """
    ## Giới thiệu đề tài
    Đề tài nhóm chúng em bao gồm 4 đề tài chính và 5 đề tài làm thêm, bao gồm:
    *  Nhận dạng khuôn mặt
    *  Xử lý ảnh
    *  Nhận dạng chữ số viết tay MNIST
    *  Nhận dạng 5 loại đối tượng trái cây dùng Yolov8
    *  Thay đổi màu nền
    *  Cân bằng màu sắc
    *  Ghi nhận di chuyển
    *  Ghi nhận cảm xúc

"""
)
