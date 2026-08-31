import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

MODEL_PATH = "next_word_model.h5"
SEQUENCE_LENGTH = 20
TEXT_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"


@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH, compile=False)

    path_to_file = tf.keras.utils.get_file("shakespeare.txt", TEXT_URL)
    with open(path_to_file, "r", encoding="utf-8") as file:
        text = file.read().lower()

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts([text])

    return model, tokenizer


def generate_text(model, tokenizer, start_text, total_words=10):
    current_text = start_text.lower().strip()
    for _ in range(total_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]

        if len(token_list) > SEQUENCE_LENGTH:
            token_list = token_list[-SEQUENCE_LENGTH:]
        input_for_model = np.array([token_list])
        predictions = model.predict(input_for_model, verbose=0)
        best_index = int(np.argmax(predictions[0]))
        next_word = tokenizer.index_word.get(best_index, "")

        if not next_word:
            break

        current_text = current_text + " " + next_word

    return current_text

st.title("Shakespeare-style Next Word Predictor")
st.write(
    "Enter a phrase and the model will continue it using the trained SimpleRNN model."
)

with st.spinner("Loading model and vocabulary..."):
    model, tokenizer = load_model_and_tokenizer()

sentence = st.text_input(
    "Starting sentence",
    placeholder="Type a phrase here...",
)

word_count = st.slider(
    "How many words to generate?", min_value=1, max_value=20, value=10
)

if st.button("Generate"):
    if sentence.strip():
        with st.spinner("Predicting the next words..."):
            result = generate_text(model, tokenizer, sentence, word_count)
        st.success("Generated text:")
        st.write(result)
    else:
        st.warning("Please enter a sentence before generating text.")
