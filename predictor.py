import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore

class ImagePredictor:
    # Constructor to ImagePredictor
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(model_path)    

    # Function to load and preprocess image to scale
    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img) 
        
        # Rescale the image to 224x224 pixel
        img = img / 224.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    # Predict the image with selected model
    def make_prediction(self, image):
        # Get the prediction label and score
        prediction = self.model.predict(image)[0][0]
        
        # Label the image with 0.5 treshold
        label = 'porn' if prediction >= 0.5 else 'neutral' 
        return label, prediction

    # Call both functions and return the prediction
    def predict_image(self, image_path):
        # Preprocess and predict the image
        preprocessed_image = self.preprocess_image(image_path)
        label, score = self.make_prediction(preprocessed_image)
        
        data = {"image_label": label, "image_score": score}
        return data