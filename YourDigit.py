import streamlit as st
from streamlit_drawable_canvas import st_canvas
from src.EasyNN import FeedForwardNeuralNetwork
import pandas as pd 
import numpy as np
import pickle
import os
import s3fs




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
    page_title="Your digit",
    page_icon="🧠",
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

    true_digit = st.slider("Please select the true digit before predict : ",0,9)
    button = st.button("Predict")
with col2:

    if canvas.image_data is not None:

        img = canvas.image_data[:, :, 0]

        # convert img 280*280 to 28*28
        img = img.reshape(28, 10, 28, 10).mean(axis=(1, 3))
        img = img.astype(int)

        # convert img matrix to vector 
        img = img.reshape(1, 28*28)
        img = pd.DataFrame(img)

        column_names = [f"pix_{i+1}" for i in range(img.shape[1])]
        data_log = img.copy()
        data_log.columns = column_names
        data_log["digit"] = true_digit

        img = img / 255

        # make prediction
        pred_prob = model.predict_prob(img)

        st.write("**Softmax outputs (probabilities) :**")
        st.markdown("""
        <style>
        div.stProgress > div > div > div > div {
            height: 10px; 
        }
        div[data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:

            for i in range(0, 5):
                if button == True:
                    prob = pred_prob[0][i, 0]
                else:
                    prob = 0
                st.markdown(f"**{i}** : {prob:.4f}%")
                st.progress(int(prob*100))

        with sub_col2:

            for i in range(5, 10):
                if button == True:
                    prob = pred_prob[0][i, 0]
                else:
                    prob = 0
                st.markdown(f"**{i}** : {prob:.4f}%")
                st.progress(int(prob*100))

if button == True:

    
    url = "https://minio.lab.sspcloud.fr/guillaume176/diffusion/ffnn_mnist/fr_mnist.parquet"
    past_log = pd.read_parquet(url)

    data_log = pd.concat([past_log, data_log])
    
    
    t1 = st.secrets["DB_1"]
    t2 = st.secrets["DB_2"]
    t3 = st.secrets["DB_3"]


    os.environ["AWS_ACCESS_KEY_ID"] = t1
    os.environ["AWS_SECRET_ACCESS_KEY"] = t2
    os.environ["AWS_SESSION_TOKEN"] = t3
    os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key = os.environ["AWS_ACCESS_KEY_ID"], 
        secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
        token = os.environ["AWS_SESSION_TOKEN"])
    

    MY_BUCKET = "guillaume176"

    FILE_PATH_OUT_S3 = f"{MY_BUCKET}/diffusion/ffnn_mnist/fr_mnist.parquet"

    with fs.open(FILE_PATH_OUT_S3, "wb") as file_out:
        data_log.to_parquet(file_out, index=False)