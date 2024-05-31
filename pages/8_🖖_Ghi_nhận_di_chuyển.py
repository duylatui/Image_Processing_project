import cv2
import streamlit as st
st.title("Moving Regconition")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)

#Lay bg frame
for i in range(10):
    _, frame = cap.read()
frame = cv2.resize(frame, (720,640))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
last_frame = gray

# Placeholder for displaying the video frames
frame_placeholder = st.empty()

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (720, 640)) 
    #Xu ly frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25,25),0) # lọc nhiễu
    abs_img = cv2.absdiff(last_frame, gray)
    # 0 - 1 =  tràn số -> Nhiễu ảnh nên ph dung abs(0-1) =1 
    last_frame = gray #gán bước ảnh htai cho last_frame

    _, img_mask = cv2.threshold(abs_img,30,255,cv2.THRESH_BINARY)
    contours , _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # trả về 1 list các contour

    for contour in contours:
        area = cv2.contourArea(contour)
        if(cv2.contourArea(contour) < 900):
            continue #bỏ qua contour có kích thước < 900

        x , y , w ,h = cv2.boundingRect(contour) #Láy ra w,h 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) # vẽ hcn
        cv2.putText(frame, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    #cv2.imshow('Window', img_mask)

    # Convert the frame to RGB and display it using Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame)
'''
_, frame = cap.read()
frame = cv2.resize(frame,(640,480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = gray

cv2.imshow('Background', background)
while True:
    _, frame = cap.read()
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break
'''