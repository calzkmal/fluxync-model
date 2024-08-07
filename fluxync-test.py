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

# Call both functions and return the prediction
def predict_image(image_path, model_path):
    preprocessed_image = preprocess_image(image_path)
    model = load_model(model_path)
    label, score = make_prediction(model, preprocessed_image)
    data = {"image_label": label, "image_score": score}
    return data

# Get available model from the 'model' directory
def get_available_model(model_dir='model'):
    model = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    return model

# Function to prompt user to choose a model
def prompt_model_selection(model):
    print("Available model:")
    for idx, model_name in enumerate(model):
        print(f"{idx + 1}. {model_name}")
    
    model_choice = int(input("Choose a model to use (by number): ")) - 1
    if model_choice < 0 or model_choice >= len(model):
        print("Invalid choice. Exiting.")
        exit(1)
    
    return model[model_choice]

# Main function to run the prediction process
def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is porn or neutral")
    parser.add_argument("image_path", type=str, help="The path to the image file")
    args = parser.parse_args()
    
    available_model = get_available_model()
    if not available_model:
        print("No model found in the 'model' directory.")
        exit(1)
    
    selected_model = os.path.join('model', prompt_model_selection(available_model))
    result = predict_image(args.image_path, selected_model)
    print(f"The model predicts this image as: {result['image_label']} with a score of {result['image_score']:.2f}")

if __name__ == "__main__":
    main()