import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64

st.set_page_config(page_title="Xử lí ảnh")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def invert_image(image):
    inverted_image = 255 - image
    return inverted_image

def apply_logarithm(image, c):
    log_image = c * np.log(1 + image)
    log_image = np.uint8(log_image)
    return log_image

def apply_power(image, gamma):
    power_image = np.power(image, gamma)
    power_image = np.uint8(power_image)
    return power_image

def apply_piecewise_linear(image, x_min, x_max, y_min, y_max):
    alpha = (y_max - y_min) / (x_max - x_min)
    beta = y_min - alpha * x_min
    linear_image = np.clip(alpha * image + beta, 0, 255)
    linear_image = np.uint8(linear_image)
    return linear_image

def equalize_histogram(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def equalize_color_histogram(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return equalized_image

def local_histogram(image, block_size, constant):
    imgb = image[:,:,0]
    imgg = image[:,:,1]
    imgr = image[:,:,2]
    
    clahe = cv2.createCLAHE(clipLimit=constant, tileGridSize=(block_size, block_size))
    cllb = clahe.apply(imgb)
    cllg = clahe.apply(imgg)
    cllr = clahe.apply(imgr)

    local_hist_image = np.dstack((cllb,cllg,cllr))
    return local_hist_image

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    st.line_chart(hist)

def HistEqualColor(imgin):
    B = imgin[:,:,0]
    G = imgin[:,:,1]
    R = imgin[:,:,2]
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    imgout = np.array([B, G, R])
    imgout = np.transpose(imgout, axes = [1,2,0]) 
    return imgout

def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.ones((m,n))
    w = w/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout

def MedianFilter(imgin):
    imgout = cv2.medianBlur(imgin, 7)
    return imgout

def Sharpen(imgin):
    # Đạo hàm cấp 2 của ảnh
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)

    # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
    # cho bộ lọc có số -4 chính giữa
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout
 
def Gradient(imgin):
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
    gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

L = 256
def CreateMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = T*np.cos(phi)
                IM = -T*np.sin(phi)
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H

def CreateMotionNoise(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)

    H = CreateMotionfilter(M, N)

    G = F*H

    G = np.fft.ifftshift(G)

    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    H = np.zeros((P,Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u-P//2)**2 + (v-Q//2)**2)
            if Duv > 0:
                H[u,v] = 1.0/(1.0 + np.power(D0/Duv,2*n))
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    gp = g[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    imgout = gp[0:M,0:N]
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def CreateNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P,Q), np.float32)
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            H[u,v] = h
    return H

def DrawNotchRejectFilter():
    H = CreateNotchRejectFilter()
    H = H*(L-1)
    H = H.astype(np.uint8)
    plt.imshow(H, interpolation='nearest')
    st.pyplot()
    
def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    H = CreateNotchRejectFilter()
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    gp = g[:,:,0]
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    imgout = gp[0:M,0:N]
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def CreateInverseMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DenoiseMotion(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)

    H = CreateInverseMotionfilter(M, N)

    G = F*H

    G = np.fft.ifftshift(G)

    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def ConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d thanh phan lien thong' % (dem-1) 
    print(text)

    a = np.zeros(dem, np.int64)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(1, dem):
        print('%4d %10d' % (r, a[r]))
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label

def CountRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem-1) 
    print(text)
    a = np.zeros(dem, np.int64)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(0, dem):
        print('%4d %10d' % (r, a[r]))

    max = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max:
            max = a[r]
            rmax = r

    xoa = np.array([], np.int64)
    for r in range(1, dem):
        if a[r] < 0.5*max:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x,y] = 0
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label

IMAGE_FUNCTIONS = {
    'Âm ảnh (Negative)': invert_image,
    'Logarithm': apply_logarithm,
    'Lũy thừa': apply_power,
    'Biến đổi tuyến tính từng phần': apply_piecewise_linear,
    'Cân bằng Histogram của ảnh màu': equalize_color_histogram,
    'Local Histogram': local_histogram,
    'Thống kê histogram': calculate_histogram,
    'Cân bằng Histogram': HistEqualColor,
    'Lọc Box' : BoxFilter,
    'Lọc Median' : MedianFilter,
    'Sharpen' : Sharpen,
    'Gradient' : Gradient,
    'Tạo nhiễu chuyển động' : CreateMotionNoise,
    'Highpass Filter' : FrequencyFilter,
    'Vẽ bộ lọc Notch Reject' : DrawNotchRejectFilter,
    'Xóa nhiễu moire' : RemoveMoire,
    'Gỡ nhiễu' : DenoiseMotion,
    'Connected Component' : ConnectedComponent,
    'Đếm hạt gạo' : CountRice
}

st.sidebar.header("Xử lí ảnh")
st.subheader('Xử lí ảnh')

uploaded_file = st.file_uploader('Chọn một ảnh', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image_histo = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    imgin = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    
    st.subheader('Ảnh gốc')
    st.image(image, channels='BGR')
    
    selected_function = st.selectbox('Chọn chức năng', list(IMAGE_FUNCTIONS.keys()))
    
    if selected_function != 'Âm ảnh (Negative)':
        if selected_function == 'Logarithm':
            c = st.slider('Hệ số c', 0.1, 10.0, 1.0)
        elif selected_function == 'Lũy thừa':
            gamma = st.slider('Số mũ gamma', 0.1, 10.0, 1.0)
        elif selected_function == 'Biến đổi tuyến tính từng phần':
            x_min = st.slider('x_min', 0, 255, 0)
            x_max = st.slider('x_max', 0, 255, 255)
            y_min = st.slider('y_min', 0, 255, 0)
            y_max = st.slider('y_max', 0, 255, 255)
        elif selected_function == 'Local Histogram':
            block_size = st.slider('Kích thước block', 3, 21, 7, step=2)
            constant = st.slider('Giá trị hằng số', 0.0, 10.0, 1.0)
        
        if selected_function == 'Logarithm':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, c)
        elif selected_function == 'Lũy thừa':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, gamma)
        elif selected_function == 'Biến đổi tuyến tính từng phần':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, x_min, x_max, y_min, y_max)
        elif selected_function == 'Local Histogram':
            processed_image = IMAGE_FUNCTIONS[selected_function](image_histo, block_size, constant)
        elif selected_function == 'Thống kê histogram':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Lọc Box':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Lọc Median':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Gradient':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Sharpen':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Tạo nhiễu chuyển động':
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        elif selected_function == 'Highpass Filter':
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        elif selected_function == 'Vẽ bộ lọc Notch Reject':
            processed_image = IMAGE_FUNCTIONS[selected_function]
        elif selected_function == 'Xóa nhiễu moire':
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        elif selected_function == 'Gỡ nhiễu':
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        elif selected_function == 'Connected Component':
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        elif selected_function == 'Cân bằng Histogram':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        elif selected_function == 'Cân bằng Histogram của ảnh màu':
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        else:
            processed_imgin = IMAGE_FUNCTIONS[selected_function](imgin)
        
        if selected_function != 'Thống kê histogram' and selected_function != 'Vẽ bộ lọc Notch Reject':
            st.subheader('Ảnh đã xử lí')
            if selected_function == 'Tạo nhiễu chuyển động' or selected_function == 'Highpass Filter' or selected_function == 'Xóa nhiễu moire' or selected_function == 'Gỡ nhiễu' or selected_function == 'Connected Component' or selected_function == 'Đếm hạt gạo':
                st.image(processed_imgin)
            else:
                st.image(processed_image, clamp=True, channels='BGR')
    else:
        inverted_image = IMAGE_FUNCTIONS[selected_function](image)
        st.subheader('Ảnh âm ảnh')
        st.image(inverted_image, channels='BGR')
