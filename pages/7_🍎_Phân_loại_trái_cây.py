import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.utils import img_to_array
from keras.utils import load_img
#from keras.preprocessing.image import  load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

#model = load_model('FV.h5')
model = load_model( os.path.join(current_dir, '../pages/20132213_20133120Fruit.h5'))
labels = {0: 'apple', 1: 'banana', 2: 'kiwi', 3: 'lemon', 19: 'mango'}

fruits = ['Apple', 'Banana',  'Kiwi', 'Lemon', 'Mango']



def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_image_path  = os.path.join(current_dir, '../pages/upload_images/' +  img_file.name)
        
        #save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        # if st.button("Predict"):
        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            '''
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            
            '''
            st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')


run()
