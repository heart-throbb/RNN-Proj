# Temperature Forecasting with SimpleRNN

This project predicts the next temperature value from the previous 30 temperature readings using a SimpleRNN neural network. The model is trained on the Jena Climate dataset and deployed with a Streamlit app for interactive forecasting.

## Project Overview

The goal of this project is to build a time-series forecasting model that learns temperature patterns from historical weather data. The workflow includes data loading, feature selection, scaling, sequence generation, SimpleRNN model training, model saving, prediction testing, and Streamlit deployment.

The app asks the user to enter the previous 30 temperature values in Celsius and returns the predicted next temperature.

## Project Files

- `1-TemperatureForecasting.ipynb` - Main notebook for loading the dataset, preprocessing data, creating sequences, training the SimpleRNN model, and saving the model/scaler.
- `2-Prediction.ipynb` - Notebook for loading the saved model and scaler, then testing single-step and multi-step temperature forecasts.
- `3-app.py` - Streamlit web app for interactive next-temperature prediction.
- `model.keras` - Trained Keras SimpleRNN model.
- `scaler.pkl` - Saved `MinMaxScaler` used to scale and inverse-transform temperature values.
- `Dataset/jena_climate_2009_2016.csv` - Jena Climate dataset used for training and testing.
- `logs/` - TensorBoard training logs.

## Dataset

This project uses the Jena Climate dataset, which contains weather time-series records collected between 2009 and 2016. The original dataset includes multiple atmospheric features such as pressure, humidity, wind velocity, and temperature.

For this project, only the temperature column is used:

```text
T (degC)
```

The notebook uses the first `100000` temperature observations, then splits them into:

- `80000` training samples
- `20000` testing samples

## Preprocessing

The preprocessing steps are:

1. Load the climate dataset with Pandas.
2. Select the `T (degC)` temperature column.
3. Limit the data to the first `100000` observations.
4. Split the data into training and testing sets using an 80/20 split.
5. Scale temperature values using `MinMaxScaler`.
6. Create input sequences with a window size of `30`.

Each training sample contains 30 previous temperature readings, and the target is the next temperature value.

## Model Architecture

The model is built with TensorFlow/Keras:

- `SimpleRNN` layer with 64 units and `tanh` activation
- `Dropout` layer with rate `0.2`
- `Dense` layer with 32 units and `relu` activation
- `Dropout` layer with rate `0.2`
- `Dense` output layer with 1 unit

The model is compiled with:

- Optimizer: `adam`
- Loss: `mse`
- Metric: `mae`

## Training Details

Training configuration:

- Window size: `30`
- Batch size: `256`
- Epochs: `15`
- Validation split: `0.20`
- Early stopping with patience `3`
- TensorBoard logging

The trained model is saved as `model.keras`, and the scaler is saved as `scaler.pkl`.

## How to Run

Clone the repository and move into this project folder:

```bash
cd RNN/Proj5
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow streamlit numpy pandas scikit-learn matplotlib seaborn tensorboard
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

1. Start the Streamlit app.
2. Enter 30 previous temperature readings in Celsius.
3. Click `Predict Temp`.
4. View the predicted next temperature.

Example output:

```text
Predicted Temperature: 19.02 °C
```

## TensorBoard

To inspect the training logs, run:

```bash
tensorboard --logdir logs/fit
```

Then open the TensorBoard URL shown in the terminal.

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Streamlit
- TensorBoard

## Learning Outcomes

This project demonstrates:

- Time-series forecasting with SimpleRNN
- Sequence generation using a sliding window
- Feature scaling and inverse transformation
- Saving and loading Keras models
- Deploying a trained deep learning model with Streamlit

## Author

Repository owner: `heart-throbb`
