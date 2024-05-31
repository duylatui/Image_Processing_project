import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import threading

# Initialize Streamlit application
st.title("Nhận diện cảm xúc")

# Load your model and the Haar cascade
face_classifier = cv2.CascadeClassifier('D:/Emotion_detection_with_CNN-main/haarcascade_frontalface_default.xml')
classifier = load_model('D:/Emotion_detection_with_CNN-main/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Placeholder for displaying the video frames
frame_placeholder = st.empty()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to RGB and display it using Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame)

# Release the webcam when done
cap.release()
