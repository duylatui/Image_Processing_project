import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import tempfile
import os

# Streamlit interface for file upload
st.subheader('Nhận dạng khuôn mặt từ video')

# File uploader widget
uploaded_file = st.file_uploader("Chọn một file video", type=["mp4", "avi", "mov"])

# Placeholder for displaying video frames
FRAME_WINDOW = st.image([])

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    # Open the video file using OpenCV
    cap = cv.VideoCapture(video_path)

    # Load pre-trained models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file1 = os.path.join(current_dir, 'svc.pkl')
    svc = joblib.load(file1)
    mydict = ['Duy', 'Giap', 'Han', 'Phuc', 'Si']

    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    file2 = os.path.join(current_dir, 'face_detection_yunet_2023mar.onnx')
    detector = cv.FaceDetectorYN.create(file2, "", (320, 320), 0.9, 0.3, 5000)

    file3 = os.path.join(current_dir, 'face_recognition_sface_2021dec.onnx')
    recognizer = cv.FaceRecognizerSF.create(file3, "")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        if faces[1] is not None:
            for face in faces[1]:  # Process each face
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                # Draw the information on the face
                x, y, w, h = face[0:4].astype(np.int32)
                cv.putText(frame, result, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())
        FRAME_WINDOW.image(frame, channels='BGR')

    cap.release()
    cv.destroyAllWindows()

# If no file is uploaded, show a message
else:
    st.info("Vui lòng tải lên một file video để bắt đầu nhận diện khuôn mặt.")