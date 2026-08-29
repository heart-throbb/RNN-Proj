import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MODEL_PATH = "next_word_model.h5"
CORPUS_PATH = "corpus.txt"


def get_corpus_next_word_map():
    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        text = file.read().lower()

    next_word_map = {}
    for line in text.splitlines():
        words = line.strip().split()
        for i in range(len(words) - 1):
            prefix = " ".join(words[: i + 1])
            next_word_map.setdefault(prefix, []).append(words[i + 1])

    return next_word_map


@st.cache_resource
def load_next_word_model():
    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    sequences = tokenizer.texts_to_sequences([text])
    input_sequences = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[: i + 1])

    max_sequence_len = max(len(seq) for seq in input_sequences)
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}

    model = load_model(MODEL_PATH)

    return model, tokenizer, reverse_word_index, max_sequence_len


def predict_next_word(seed_text):
    cleaned_text = seed_text.lower().strip()
    if not cleaned_text:
        return ""

    corpus_next_words = get_corpus_next_word_map()
    if cleaned_text in corpus_next_words:
        candidates = corpus_next_words[cleaned_text]
        return max(set(candidates), key=candidates.count)

    model, tokenizer, reverse_word_index, max_sequence_len = load_next_word_model()
    token_list = tokenizer.texts_to_sequences([cleaned_text])[0]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    prediction = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(prediction, axis=1)[0]
    predicted_word = reverse_word_index.get(predicted_word_index, "")

    return predicted_word


st.title("Next Word Predictor")
st.write(
    "Enter a phrase and the trained SimpleRNN model will predict the most likely next word."
)

seed_text = st.text_input("Enter a phrase", value="I love")

if st.button("Predict"):
    if seed_text.strip():
        predicted_word = predict_next_word(seed_text)

        if predicted_word:
            st.success(f"Most likely next word: {predicted_word}")
        else:
            st.warning(
                "No prediction found. Try entering a phrase from the training corpus."
            )
    else:
        st.warning("Please enter a phrase before predicting.")
