# SimpleRNN Next Word Predictor

This project uses a SimpleRNN model trained on a small text corpus to predict the most likely next word in a phrase.

## Files

- `1-next_word_prediction.ipynb`: notebook that shows the training and inference workflow.
- `2-main.py`: Streamlit app for interactive prediction.
- `corpus.txt`: training text used to build the vocabulary and sequences.
- `next_word_model.h5`: saved Keras model.

## How it works

The model:

1. Reads the corpus text.
2. Tokenizes the words using `Tokenizer`.
3. Builds sequence windows from the text.
4. Pads the input sequences to a fixed length.
5. Trains a SimpleRNN to predict the next word.

At inference time, the app:

1. Accepts a phrase from the user.
2. Converts it into token IDs.
3. Keeps the most recent sequence window.
4. Sends the sequence to the trained model.
5. Returns the predicted next word.

## Requirements

Install the required packages:

```bash
pip install tensorflow numpy streamlit
```

## Run the app

From this project folder, run:

```bash
streamlit run 2-main.py
```

Then type a phrase such as:

- `I love`
- `Machine learning is`
- `Deep learning uses`

The model will return the most likely next word.

## Notes

- The model expects `next_word_model.h5` and `corpus.txt` to be in the same folder as the app.
- The input is converted to lowercase before prediction for consistency with the training data.
- This is a simple sequence-learning example that demonstrates next-word prediction with a recurrent neural network.
