# **PROJECT IDEA**

Aim of this project is to develop a binary image classification of NSFW Images based on CNN using MobileNetV2 from Keras model.
Data description:

Total images: 6400 images splitted into train-val-test with 70-20-10 percent split.
- Training: Found 4480 images belonging to 2 classes.
- Validation: Found 2560 images belonging to 2 classes.
- Test: Found 1280 images belonging to 2 classes.
- Classes: 'neutral' and 'porn'.

**REQUIREMENTS**
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/768px-NumPy_logo_2020.svg.png" width="50%" height="50%">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/768px-TensorFlow_logo.svg.png" width="50%" height="50%">
<img src="https://keras.io/img/logo.png" width="50%" height="50%">


---
**HOW TO RUN**

Open up terminal/cmd, type
```
    python fluxync.py "target image directory"
```

---
## **Development**

- Early Accuracy:
- ![image](https://github.com/user-attachments/assets/8f4eab6a-cb9d-46ef-89cc-a51f64efdfe5)

- Early Loss:
- ![image](https://github.com/user-attachments/assets/bf12c065-196d-4f09-9f3a-273203fdf8e5)

The model shows a decent performance in predicting images, with 78.2% of test accuracy.
Further development may requires an addition of epochs on training, adding more Convolutional layer on base model, feed more data to minimalize biases.

---

## **Final model stats with 95% accuracy**

I've added transfer learning method by using MobileNetV2 model by Keras.

- Accuracy:
- ![image](https://github.com/user-attachments/assets/026f2b4a-0a4e-45ac-815d-4500efcbe5d9)

- Loss:
- ![image](https://github.com/user-attachments/assets/b86de15c-ac2d-40fc-b8dd-c9a7e19a08d4)

- Confusion Matrix:
- ![image](https://github.com/user-attachments/assets/13f5aa1c-a468-4cd5-84e5-5d579a9e83a8)

The model shows a superb performance with 95.3% test accuracy.
Further development may require usage of different Keras model, or adding more images to training data.
