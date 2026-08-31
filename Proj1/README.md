# SimpleRNN IMDB Sentiment Analysis

This project builds a movie-review sentiment classifier with TensorFlow and Keras using the IMDB dataset. It includes an embedding walkthrough, SimpleRNN model training, and a Streamlit interface for predicting sentiment on custom reviews.

## Project files

- `1-embedding.ipynb`: introduces integer encoding, one-hot representations, and padding for text sequences.
- `2_Imdb.ipynb`: prepares the IMDB dataset and trains the SimpleRNN sentiment model.
- `3-Imdb_Prediction.ipynb`: loads the trained model and tests review predictions.
- `main.py`: Streamlit app for interactive positive/negative sentiment classification.
- `simple_rnn_imdb.h5`: saved Keras model used by the app.

## How it works

The model:

1. Loads the IMDB movie-review dataset.
2. Converts reviews into integer sequences.
3. Pads each review to a fixed length.
4. Trains a SimpleRNN model with an embedding layer.
5. Predicts whether a review is positive or negative.

At inference time, the app:

1. Accepts a user review.
2. Tokenizes and pads the input.
3. Sends the sequence to the trained model.
4. Displays the sentiment label and confidence score.

## Requirements

Install the required packages:

```bash
pip install tensorflow numpy streamlit
```

The first time the IMDB dataset is accessed, TensorFlow may download it locally. Those generated files are not required in the repository.

## Run the app

From this directory, run:

```bash
streamlit run main.py
```

Then open the local URL shown by Streamlit and enter a movie review. The app pads the review to 500 tokens and displays the predicted sentiment and score.

## Example

- Input: "This movie was amazing and very entertaining."
- Output: Positive sentiment with a probability score.

## Notes

- The application expects `simple_rnn_imdb.h5` in the same directory as `main.py`.
- Unknown words use the IMDB dataset's unknown-word index.
