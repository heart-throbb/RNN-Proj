# SMS Spam Detection with SimpleRNN

This project builds an SMS spam classifier using a SimpleRNN neural network. The model is trained on the SMS Spam Collection dataset and deployed through a small Streamlit app where users can enter a message and get a spam probability.

## Project Overview

The goal of this project is to classify SMS messages as:

- `ham` - normal message
- `spam` - unwanted promotional or scam message

The workflow includes text preprocessing, tokenization, sequence padding, model training, model saving, prediction, and Streamlit deployment.

## Project Files

- `1-SMSSpamDetection.ipynb` - Main notebook for loading the dataset, preprocessing text, training the SimpleRNN model, evaluating it, and saving the model/tokenizer.
- `2-Prediction.ipynb` - Notebook for loading the saved model and tokenizer, then testing predictions on new SMS messages.
- `3-app.py` - Streamlit web app for interactive spam detection.
- `model.keras` - Trained Keras model.
- `tokenizer.pkl` - Saved tokenizer used to convert text into numeric sequences.
- `Dataset/SMSSpamCollection` - SMS Spam Collection dataset.
- `Dataset/readme` - Original dataset description, usage notes, and license/disclaimer.

## Dataset

This project uses the SMS Spam Collection v1 dataset. It contains 5,574 English SMS messages labeled as `ham` or `spam`.

Dataset distribution:

- 4,827 ham messages
- 747 spam messages

Each row contains a label and the raw SMS text.

Example format:

```text
ham     Ok lar... Joking wif u oni...
spam    Free entry in 2 a wkly comp to win FA Cup final tkts...
```

## Model Architecture

The model is a binary text classification network built with TensorFlow/Keras:

- Embedding layer
- SimpleRNN layer with 64 units
- Dropout layer
- Dense hidden layer with ReLU activation
- Dropout layer
- Dense output layer with sigmoid activation

The sigmoid output gives a probability score between 0 and 1. A threshold of `0.5` is used:

- Probability `>= 0.5` means spam
- Probability `< 0.5` means ham

## Preprocessing

Before prediction, each SMS message is cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing special characters
- Removing extra spaces
- Tokenizing text
- Padding or truncating sequences to a fixed length of `100`

## Training Details

The training notebook uses:

- `MAX_WORDS = 10000`
- `MAX_LEN = 100`
- Adam optimizer
- Binary cross-entropy loss
- Accuracy, precision, and recall metrics
- Early stopping
- TensorBoard logging
- Class weights to help with the imbalanced dataset

Recorded test evaluation from the notebook:

- Test loss: `0.1372`
- Test accuracy: `0.9642`

## How to Run

Clone the repository and move into this project folder:

```bash
cd RNN/Proj4
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install tensorflow streamlit numpy pandas scikit-learn
```

Run the Streamlit app:

```bash
streamlit run 3-app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Usage

Example spam-like message:

```text
Congratulations! You won a free prize. Claim now!
```

Example ham-like message:

```text
Hey, are we still meeting at 6 today?
```

The app returns whether the message is likely spam or normal, along with the spam probability.

## Notes

- Keep `model.keras` and `tokenizer.pkl` in the same folder as `3-app.py`.
- The model is trained for learning/demo purposes and should not be used as a production spam filter without more validation.
- The dataset may contain real-world SMS-style abbreviations, numbers, and informal text.

## Tech Stack

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- TensorBoard

## Dataset Attribution

The SMS Spam Collection dataset was collected by Tiago Agostinho de Almeida and Jose Maria Gomez Hidalgo. Please refer to `Dataset/readme` for the original dataset description, citation request, and license/disclaimer.

## Author

Repository owner: `heart-throbb`
