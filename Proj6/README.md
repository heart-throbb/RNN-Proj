# Proj6 — Human Activity Recognition using Pure SimpleRNN

This project focuses on human activity recognition from smartphone sensor time-series data using a pure SimpleRNN architecture. The model is trained on accelerometer and gyroscope readings to classify daily activities such as walking, sitting, standing, and laying down.

## Project Goal

The objective of this project is to build a sequence-based classifier that learns temporal patterns from motion sensor signals over fixed time windows.

Most importantly, this project uses:

- Pure SimpleRNN
- No LSTM
- No GRU
- No CNN
- No Transformer

The goal is to keep the model simple and study how well a recurrent neural network performs on time-series activity recognition.

## Dataset Used

This project uses the UCI Human Activity Recognition Using Smartphones dataset.

The dataset contains smartphone sensor recordings from 30 subjects, sampled at 50 Hz, with fixed windows of 128 readings. The data includes accelerometer and gyroscope signals, and the target classes are the six activities listed below.

Dataset link:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

### Activities classified

- 🚶 Walking
- 🪜 Walking Upstairs
- 🪜 Walking Downstairs
- 🪑 Sitting
- 🧍 Standing
- 🛏️ Laying

## Sensor Data

The model uses time-series data from smartphone sensors, specifically:

- Accelerometer signals
- Gyroscope signals
- Temporal windows of fixed length (128 time steps)

This makes the problem a multivariate time-series classification task.

## Project Structure

- `1_HumanActivityRecognition.ipynb` — notebook for data preparation, modeling, training, and evaluation
- `2-Prediction.ipynb` — notebook for inference using the trained model
- `3-app.py` — application script for running predictions through the trained model
- `Dataset/` — dataset folder used for training and evaluation
- `model.keras` — trained SimpleRNN model
- `scaler.pkl` — scaler used for data normalization
- `label_encoder.pkl` — label encoder used to map class labels to integers
- `logs/` — training logs and saved experiment outputs

## Model Architecture

The core model is a sequential recurrent neural network built using SimpleRNN layers. The network learns patterns over time from sensor windows and outputs one of the six activity classes.

Typical architecture idea:

- Input shape: time steps × sensor features
- SimpleRNN layers
- Dense layers
- Softmax output layer for multi-class classification

This project intentionally avoids more advanced architectures so the focus remains on evaluating the strength of SimpleRNN for activity recognition.

## Requirements

Python environment requirements typically include:

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Joblib

You can install them using:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Open the training notebook

Run the notebook:

```bash
1_HumanActivityRecognition.ipynb
```

This notebook covers:

- loading the dataset
- preprocessing the sensor windows
- scaling/normalizing features
- training the SimpleRNN model
- evaluating performance
- saving the trained model and preprocessing artifacts

### 2. Run prediction notebook

Open:

```bash
2-Prediction.ipynb
```

Use it to load the trained model and test predictions on sample windows.

### 3. Run the app

To launch the application:

```bash
python 3-app.py
```

This app loads the saved model and allows prediction using the trained activity-recognition pipeline.

## Training Notes

The project is designed around sequential learning from human motion data. For each sensor window:

- features are extracted from accelerometer and gyroscope values
- the sequence is passed into the SimpleRNN model
- the model predicts the corresponding activity label

Because the data is time-dependent, recurrent networks are a natural fit for this task.

## Expected Outcome

The final model should be able to classify human activities based on short windows of smartphone motion data with good accuracy for the six classes above.

## Summary

This project demonstrates a practical implementation of activity recognition using:

- smartphone motion sensor data
- fixed-length temporal windows
- pure SimpleRNN modeling
- multi-class classification for human activities

It serves as a strong example of applying recurrent neural networks to real-world time-series sensor data.

## Reference

- UCI Human Activity Recognition Using Smartphones dataset
- https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Author

Repository owner: `heart-throbb`
