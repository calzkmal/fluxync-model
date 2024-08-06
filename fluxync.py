import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Function to load and preprocess image to scale
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 224.0  # Rescale
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict the image with selected model
def make_prediction(model, image):
    prediction = model.predict(image)[0][0]  # Get the prediction
    label = 'porn' if prediction >= 0.5 else 'neutral'
    return label, prediction

# Call both functions and print out the prediction
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    model = load_model('fluxync.keras')
    label, score = make_prediction(model, preprocessed_image)
    print(f"The model predicts this image as: {label} with a score of {score:.2f}")
    
    data = {
        "image_label": label,
        "image_score": score
    }
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether an image is porn or neutral")
    parser.add_argument("image_path", type=str, help="The path to the image file")
    args = parser.parse_args()
    predict_image(args.image_path)
