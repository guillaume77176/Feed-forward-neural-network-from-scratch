import streamlit as st
from streamlit_drawable_canvas import st_canvas
from src.EasyNN import FeedForwardNeuralNetwork
import pandas as pd 
import numpy as np
import pickle


@st.cache_resource
def init_model():

    # arbitrary data to init the model
    X = np.array([[1, 0, 1], [7, 8, 2]])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(1, 10)

    model = FeedForwardNeuralNetwork(X, y, "cross_entropy", 0.01, 10)

    with open("mnist_ffnn.pkl", "rb") as file:
        loaded_params = pickle.load(file)

    model.pre_train(loaded_params)

    return model


model = init_model()
    

st.set_page_config(
    page_title="Ma app",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)


st.title("Try using the neural network to recognise the digit written by hand below!")


col1, col2 = st.columns(2)  # 2 colonnes égales

with col1:
    # main canvas
    canvas = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    button = st.button("Predict")

with col2:

    if canvas.image_data is not None and button is True:

        img = canvas.image_data[:, :, 0]

        # convert img 280*280 to 28*28
        img = img.reshape(28, 10, 28, 10).mean(axis=(1, 3))
        img = img.astype(int)

        # convert img matrix to vector 
        img = img.reshape(1, 28*28)
        img = pd.DataFrame(img)
        img = img / 255

        # make prediction
        pred_prob = model.predict_prob(img)
        pred = np.argmax(pred_prob[0])

        st.write("PREDICTED DIGIT")
        st.write(f"<h1 style='font-size:100px'>{pred}</h1>", unsafe_allow_html=True)