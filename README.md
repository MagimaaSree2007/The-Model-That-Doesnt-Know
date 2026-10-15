# The-Model-That-Doesnt-Know
Image classification system with uncertainty detection to identify known and unfamiliar clothing images
# The Model That Doesn't Know

## 1. Project Overview

The Model That Doesn't Know is an image classification system designed to recognize familiar clothing categories and reject uncertain or unfamiliar inputs as **UNKNOWN**.

The project uses a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset and a confidence-threshold mechanism for uncertainty handling.

## 2. Problem Statement

Traditional image classification models may confidently assign an input to a class even when the image does not belong to any of the categories learned during training. This project aims to reduce such errors by allowing the model to reject predictions when its confidence is below a selected threshold.

## 3. Dataset

The project uses the Fashion-MNIST dataset.

**Known classes:**
- T-shirt/top
- Trouser
- Dress
- Sandal
- Sneaker

**Unknown classes excluded from training:**
- Pullover
- Coat
- Shirt
- Bag
- Ankle boot

The five known classes are used to train the classifier. Images from the excluded classes are used to evaluate unknown-image detection.

## 4. Preprocessing

- Selected the five known classes for training.
- Remapped their labels to values from 0 to 4.
- Normalized pixel values to the range 0–1.
- Resized the input representation to 28 × 28 pixels and used a single grayscale channel.
- Split the known training data into training and validation sets.

## 5. Model Architecture

A CNN was developed using TensorFlow/Keras with:
- Two convolutional layers
- Two max-pooling layers
- A flatten layer
- A fully connected dense layer
- Dropout for regularization
- A five-class softmax output layer

The model was trained using the Adam optimizer and sparse categorical cross-entropy loss.

## 6. Uncertainty Detection

The model calculates class probabilities for an input image and identifies the maximum predicted probability.

The selected confidence threshold is 0.99 (99%).

- If the maximum probability is at least 99%, the predicted class is accepted.
- If the maximum probability is below 99%, the system returns UNKNOWN.

This threshold is a heuristic and does not guarantee that every accepted prediction is correct or every unfamiliar image is rejected.

## 7. Evaluation

The model achieved approximately 97.80% accuracy on the held-out known-class test images before applying rejection.

Using the 0.99 confidence threshold, the recorded test results were:

- Known-image acceptance rate: 82.58%
- Known-image rejection rate: 17.42%
- Unknown-image detection rate: 81.20%
- Unknown-image incorrect acceptance rate: 18.80%
- Balanced accuracy for known acceptance and unknown detection: 81.89%

These results indicate that the model can reject many unfamiliar-category images, but some known images are also rejected and some unknown images are accepted.

## 8. Observations

- The model performs best on images similar to the Fashion-MNIST dataset.
- Real-world clothing photographs may produce lower confidence because their appearance differs from the training images.
- A high confidence threshold can reject uncertain predictions but may also reject correct classifications.
- Softmax confidence is not a guarantee of prediction correctness.

## 9. Limitations

- The model is trained on small, grayscale Fashion-MNIST images.
- Real-world photographs may differ in background, lighting, color, scale and orientation.
- The confidence threshold may require recalibration for different datasets.
- The system does not guarantee detection of every out-of-distribution image.

## 10. Future Improvements

- Test on more diverse real-world images.
- Explore out-of-distribution detection methods beyond maximum softmax probability.
- Calibrate confidence scores using validation data.
- Compare different uncertainty detection methods.
- Improve robustness to image quality and background changes.

## 11. Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Streamlit
- Pillow
- Scikit-learn
- Matplotlib

## 12. How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Ensure that `model.weights.h5` is in the same directory as `app.py`.

## 13. Conclusion

The project demonstrates an image classification system that can recognize five known clothing categories and reject uncertain predictions as UNKNOWN. It also highlights the limitations of confidence-threshold-based uncertainty detection and the challenges of applying a model trained on a controlled dataset to real-world images.
