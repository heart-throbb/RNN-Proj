# Next Word Predictor with SimpleRNN

This project demonstrates a next-word prediction model using a SimpleRNN architecture. It is trained on Shakespeare text and can generate text continuations based on a user-provided prompt.

## Project Files

- `1_next_word_predictor.ipynb` - Notebook containing the full model training and prediction logic.
- `2-app.py` - Streamlit web app for interacting with the trained model.
- `next_word_model.h5` - Trained Keras model saved in HDF5 format.

## What the Model Does

The model learns word sequences from Shakespeare text and predicts the most likely next word based on the current sentence context.

It uses:

- Tokenization
- Sequence generation
- Embedding layer
- SimpleRNN layer
- Dense output layer with softmax activation

## How to Run the Streamlit App

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install tensorflow numpy streamlit
```

4. Run the app:

```bash
streamlit run 2-app.py
```

5. Open the local URL shown in the browser.

## Example Usage

- Input: `to be or not to be`
- Output: `to be or not to be a king of mine own hands the king's name is it not a man to me and i am not`

## Notes

- The model expects the saved file `next_word_model.h5` to be present in the same folder.
- The app downloads the Shakespeare dataset automatically if it is not already available on the system.
- The app uses a fixed sequence length of 20 words for prediction.

## Requirements

- Python 3.9+
- TensorFlow
- NumPy
- Streamlit

## Summary

This project is a small but practical example of language modeling with recurrent neural networks, showing how text generation can be deployed as a simple web application.
