from predictor import ImagePredictor
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/detect-image", methods=['POST'])
def predict_image():
    data = request.get_json()
    if 'image_path' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    image_path = data['image_path']

    # Model config
    model_path = 'model/fluxync-mobilenetv2.keras'
    predictor = ImagePredictor(model_path)

    result = predictor.predict_image(image_path)

    return jsonify({"image_label": result['image_label'],
                    "image_score": result['image_score']
                    })

app.run(host='0.0.0.0', port=8888)