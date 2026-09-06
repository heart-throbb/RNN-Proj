import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

WINDOW_SIZE = 30


@st.cache_resource
def load_resources():
    model = load_model("model.keras")
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, scaler


model, scaler = load_resources()


def predict_temperature(temperatures):
    values = np.array(temperatures).reshape(-1, 1)
    scaled = scaler.transform(values)
    sequence = scaled[-WINDOW_SIZE:]
    sequence = sequence.reshape(1, WINDOW_SIZE, 1)
    prediction_scaled = model.predict(sequence, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]


st.title("Temperature Forecasting")
st.divider()
st.subheader("Enter the previous 30 temperatures")
temperatures = []

for i in range(WINDOW_SIZE):
    value = st.number_input(f"Temperature {i + 1}", value=20.0, step=0.1, format="%.2f")
    temperatures.append(value)

if st.button("Predict Temp", use_container_width=True):
    prediction = predict_temperature(temperatures)
    st.success(f"Predicted Temperature: " f"**{prediction:.2f} °C**")
