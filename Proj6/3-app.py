import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


@st.cache_resource
def load_resources():
    model = load_model("model.keras")
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    return (model, scaler, label_encoder)


model, scaler, label_encoder = load_resources()

activity_names = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

st.title("Human Activity Recognition")

st.divider()

st.subheader("Upload Sensor Feature CSV")
uploaded_file = st.file_uploader("Upload a CSV containing 561 features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    st.write("Uploaded Data:")
    st.dataframe(data.head())
    if data.shape[1] != 561:
        st.error(f"Expected 561 features, " f"but received {data.shape[1]}.")
    else:
        data_scaled = scaler.transform(data)
        data_rnn = data_scaled.reshape(data_scaled.shape[0], 561, 1)
        probabilities = model.predict(data_rnn, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        st.subheader("Predictions")
        for i, prediction in enumerate(predictions):
            class_id = label_encoder.inverse_transform([prediction])[0]
            activity = activity_names[class_id]
            confidence = probabilities[i][prediction]
            st.success(f"Sample {i + 1}: " f"{activity} " f"({confidence * 100:.2f}%)")
