import streamlit as st
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100


@st.cache_resource
def load_resources():
    model = load_model("model.keras")
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


model, tokenizer = load_resources()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_message(message):
    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    probability = model.predict(padded, verbose=0)[0][0]
    return probability


st.title("SMS Spam Detector")
st.divider()

message = st.text_area(
    "Enter your SMS message:",
    placeholder=("Example: " "Congratulations! You won a free prize!"),
    height=150,
)

if st.button("Predict", use_container_width=True):
    if not message.strip():
        st.warning("Please enter an SMS message.")
    else:
        probability = predict_message(message)
        if probability >= 0.5:
            st.error("SPAM MESSAGE!!!!")
            st.write(f"Spam Probability: " f"**{probability:.2%}**")
        else:
            st.success("NORMAL MESSAGE...")
            st.write(f"Spam Probability: " f"**{probability:.2%}**")
