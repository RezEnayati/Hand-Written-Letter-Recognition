# Live Digit Recognition Using CNN 

This project is an interactive digit classifier that uses a Convolutional Neural Network (CNN) trained on the MNIST dataset. It features a sketchpad interface built with [Gradio](https://www.gradio.app/) and is deployed live on Hugging Face Spaces.

[![Gradio](https://img.shields.io/badge/Gradio-App-blue?logo=gradio)](https://gradio.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
---
## Features: 
- **Deep CNN Architecture** with BatchNormalization and Dropout for robust performance.
- **Real-Time Inference** using a Gradio sketchpad for interactive predictions.
- **Data Augmentation** during training to improve generalization on messy, unseen input.
- **Deployed on Hugging Face Spaces** for live demonstration and ease of access.
--- 
## Live Demo: 
[Try it out on Hugging Face Spaces](https://huggingface.co/spaces/rezaenayati/Live_Digit_Recognition_Using_CNN)

Just draw a digit and hit Submit.  
The model will return the top 3 guesses along with their confidence scores.

---
## How it works: 
1. The model was trained on the MNIST dataset, a collection of 28×28 grayscale images of handwritten digits. Each pixel value represents the intensity of the digit stroke, with darker pixels indicating higher stroke intensity. The dataset includes 60k training examples and 10k test examples. 
2. After loading the data, the values in the dataset were normalized simply by dividing each digit in the dataset by 255.0 to get the values in a range of [0, 1]. This makes gradient descent more stable and helps the network converge faster by keeping input values in a small, consistent range. I also reshaped the matrices in the training and test examples to turn them into 4D arrays, which makes them compatible with the Conv2D layers in Keras.
3. For the model architecture, I first used two **convolutional layers** with **32** and **64** units respectively, using the **ReLU** activation with a window size of **(3,3)**. After each of the convolutional layers, I did **MaxPooling** with a pool size of **(2,2)**.

After the convolutional layers, I had to flatten the data using the **`Flatten()`** method in the model to be able to feed the training examples into the single **Dense** hidden layer that I had, with **128** activation units still using **ReLU**.

The output layer was a **Dense** layer with **10** units corresponding to digits **0–9**, with a **linear** activation. I didn't use a traditional **Softmax** to get rid of intermediate values and instead ran a **softmax function after the model has made a prediction**.




