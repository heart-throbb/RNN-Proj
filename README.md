# SimpleRNN IMDB Sentiment Analysis

This project builds a movie-review sentiment classifier with TensorFlow and Keras using the IMDB dataset. It includes an embedding walkthrough, SimpleRNN model training, and a Streamlit interface for making predictions on custom reviews.

## Contents

- `1-embedding.ipynb`: introduces integer encoding, one-hot representations, and padding for text sequences.
- `2_Imdb.ipynb`: prepares the IMDB dataset and trains the SimpleRNN sentiment model.
- `3-Imdb_Prediction.ipynb`: loads the trained model and tests review predictions.
- `main.py`: Streamlit application for interactive positive/negative sentiment classification.
- `simple_rnn_imdb.h5`: saved Keras model used by the application.
- `simple_rnn_imdb/`: local generated/runtime files, intentionally excluded from Git.

## Requirements

Install the dependencies from the project requirements file if available, or install the main packages directly:

```bash
pip install tensorflow numpy streamlit
```

The first TensorFlow IMDB dataset access may download dataset files locally. Those generated files are not required in the repository.

## Run the application

From this directory, run:

```bash
streamlit run main.py
```

Then open the local URL shown by Streamlit and enter a movie review. The app pads the tokenized review to 500 tokens and displays the predicted sentiment and score.

## Notes

- The application expects `simple_rnn_imdb.h5` in the same directory as `main.py`.
- Unknown words use the IMDB dataset's unknown-word index.
