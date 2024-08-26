from predictor import ImagePredictor

if __name__ == "__main__":
    # Model config
    model_path = 'model/fluxync-mobilenetv2.keras'
    predictor = ImagePredictor(model_path)

    # Image config
    image_path = 'test_images/porn/porn-4.jpg'
    result = predictor.predict_image(image_path)

    print(f"This model predicts the image as: {result['image_label']} with a score of {result['image_score']}")