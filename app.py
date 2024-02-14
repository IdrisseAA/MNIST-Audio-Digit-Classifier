import os
import numpy as np
from flask import Flask, jsonify
from keras.models import load_model
from src.build_features import prepare_audio_features_for_prediction

# Defining some constants
MODELS_FOLDER = './models'
PRODUCTION_FOLDER = './data/production_data'
app = Flask(__name__)


# Function to load a model and make a prediction
def predict_with_model(model_path, audio_features):
    model = load_model(model_path)
    prediction = model.predict(audio_features)
    predicted_digit = str(np.argmax(prediction))
    prediction_probabilities = prediction
    return predicted_digit, prediction_probabilities


# Function to predict using a specified model
def predict_function(model_type, audio_file):
    audio_file_path = os.path.join(PRODUCTION_FOLDER, audio_file)
    audio_features = prepare_audio_features_for_prediction(audio_file_path=audio_file_path)
    model_path = f'{MODELS_FOLDER}/{model_type}.keras'

    try:
        predicted_digit, prediction_probabilities = predict_with_model(model_path, audio_features)
        confidence_level = f"{round(np.max(prediction_probabilities) * 100, 3)} %"

        # Convert prediction_probabilities to list if it's not already
        prediction_probabilities = prediction_probabilities[0].tolist()

        all_result = {}
        for i, p in enumerate(prediction_probabilities, start=0):
            max_confidence = round(np.max(p) * 100, 3)
            all_result[i] = f'for {max_confidence} %'

        # print(all_result)
        # Preparing the output
        output = {
            'Predicted_Digit': predicted_digit,
            'Confidence_Level': confidence_level,
            'All_Result': all_result
        }

        return jsonify(output)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404  # Return 404 Not Found for missing model file

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return 500 Internal Server Error for other exceptions


# Endpoints
@app.route("/predict_using_cnn/<audio_file>")
def predict_using_cnn(audio_file):
    return predict_function('cnn', audio_file)


@app.route("/predict_using_lstm/<audio_file>")
def predict_using_lstm(audio_file):
    return predict_function('lstm', audio_file)


@app.route("/predict_using_conv1d/<audio_file>")
def predict_using_conv1d(audio_file):
    return predict_function('conv1d', audio_file)


@app.route("/predict_using_hybrid/<audio_file>")
def predict_using_hybrid(audio_file):
    return predict_function('hybrid', audio_file)


# Welcome page
@app.route("/")
def welcome_function():
    return "Welcome to MNIST Audio Digit Classifier"


if __name__ == "__main__":
    app.run()
