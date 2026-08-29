# Next Word Predictor

This project uses a SimpleRNN model trained on a small text corpus to predict the most likely next word in a phrase.

## Files

- `corpus.txt`: training text used to build the vocabulary and sequences.
- `next_word_model.h5`: saved Keras model.
- `next_word_prediction.ipynb`: notebook that shows the training and inference workflow.
- `main.py`: Streamlit app for interactive prediction.

## How it works

The model:

1. Reads the corpus text.
2. Tokenizes the words using `Tokenizer`.
3. Builds n-gram sequences from the text.
4. Pads the input sequences to a fixed length.
5. Trains a SimpleRNN to predict the next word.

At inference time, the app:

1. Accepts a phrase from the user.
2. Converts it into token IDs.
3. Pads the sequence to the saved training length.
4. Uses the model to predict the next word.

## Requirements

Use the project virtual environment or install the required packages:

```bash
pip install tensorflow numpy streamlit
```

## Run the app

From this folder, run:

```bash
streamlit run main.py
```

Then type a phrase such as:

- `I love`
- `Machine learning is`
- `Deep learning uses`

The model will return the most likely next word.

## Notes

- The model expects the same corpus and vocabulary setup used during training.
- The app uses `next_word_model.h5` and `corpus.txt` from the same folder.
- The input is converted to lowercase before prediction for consistency with the training data.
