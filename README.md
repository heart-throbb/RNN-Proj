# SimpleRNN Projects Repository

A curated collection of hands-on deep learning projects built with Python, TensorFlow/Keras, and recurrent neural networks. This repository focuses on the SimpleRNN architecture and demonstrates how it can be applied to real-world sequence problems such as text classification, next-word prediction, time-series forecasting, and sensor-based activity recognition.

This folder brings together six structured learning projects and one additional IMDb-based example, each designed to help you understand how recurrent neural networks work in practice.

## Overview

SimpleRNN models are well-suited for sequential data, where understanding context and order matters. In this repository, you will explore:

- Text sentiment analysis
- Next-word prediction
- Time-series forecasting
- Spam detection
- Human activity recognition
- Sequence modeling with recurrent neural networks

## Repository Structure

| Project         | Focus                      | Description                                                                                   |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| [Proj1](Proj1/) | Sentiment Analysis         | Classifies movie reviews as positive or negative using an IMDb dataset and SimpleRNN model.   |
| [Proj2](Proj2/) | Next Word Prediction       | Trains a SimpleRNN model to predict the next word from a short text sequence.                 |
| [Proj3](Proj3/) | Text Generation            | Uses a recurrent architecture to predict the next word in a Shakespeare-inspired text corpus. |
| [Proj4](Proj4/) | SMS Spam Detection         | Identifies spam vs. ham messages using tokenized text sequences and a SimpleRNN classifier.   |
| [Proj5](Proj5/) | Temperature Forecasting    | Predicts the next temperature using windowed historical climate data.                         |
| [Proj6](Proj6/) | Human Activity Recognition | Classifies human activities from smartphone sensor time-series data using pure SimpleRNN.     |

## Project Highlights

### 1. IMDb Sentiment Analysis

This project introduces the foundations of sequence learning for text. It uses tokenized movie reviews, embedding layers, and SimpleRNN to classify whether a review is positive or negative.

### 2. Next Word Prediction

This project demonstrates how a recurrent model learns patterns from word sequences and predicts the most likely next token in a phrase.

### 3. Shakespeare Text Generation

A more language-oriented example that trains a model on text sequences and generates continuation based on a user prompt.

### 4. SMS Spam Detection

This project teaches how to process short text messages, convert them into sequence inputs, and build a binary classifier using a recurrent neural network.

### 5. Temperature Forecasting

This time-series project uses past temperature values to forecast the next value, helping you understand autoregressive sequence learning.

### 6. Human Activity Recognition

This project uses accelerometer and gyroscope signals from smartphones to classify activities such as walking, sitting, standing, and laying. It is a strong example of multivariate time-series classification using RNNs.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Matplotlib / Seaborn
- TensorBoard

## Learning Outcomes

By exploring these projects, you will learn how to:

- Prepare text and time-series data for sequence models
- Build and train SimpleRNN networks in Keras
- Use embedding layers for text representation
- Handle sequence padding and fixed-length windows
- Save and load trained models
- Deploy deep learning models with Streamlit
- Understand how recurrent networks learn temporal dependencies

## Getting Started

Each project in this repository is self-contained and includes its own notebooks, model files, and app scripts. The easiest way to start is to open the project folder that matches your learning goal and follow its README instructions.

### Example workflow

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install common dependencies
pip install tensorflow numpy pandas scikit-learn streamlit jupyterlab
```

Then move into a specific project folder and run its app or notebook:

```bash
cd Proj4
streamlit run 3-app.py
```

## Recommended Learning Path

1. Start with [Proj1](Proj1/) to understand text classification with SimpleRNN.
2. Continue with [Proj2](Proj2/) and [Proj3](Proj3/) for sequence modeling and text prediction.
3. Study [Proj4](Proj4/) and [Proj5](Proj5/) for practical classification and forecasting examples.
4. Finish with [Proj6](Proj6/) to work on real sensor-based time-series data.

## Notes

- Most projects are educational and designed to demonstrate model workflows clearly.
- Model weights and saved artifacts are included in each project folder where applicable.
- Some notebooks may require downloading the dataset the first time they are run.
- Each project folder has its own README with more detailed instructions and explanations.

## Final Note

This repository is intended as a practical learning archive for understanding recurrent neural networks, especially the SimpleRNN architecture. It is ideal for beginners and intermediate learners who want to explore how sequence models work on both text and time-series data.

---

If you are learning deep learning step by step, this collection is a strong foundation for moving from basic neural networks to more advanced recurrent and transformer-based architectures.

## Author

Repository owner: `heart-throbb`
