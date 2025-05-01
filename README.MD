# Handwritten Digit Cassifier with Gradio + TensorFlow 

In this Project I created a live handwritten letter recognition. You can draw a digit from 0-9 in the Gradio Sketchpad and get a prediction in real-time using a Convolutional Neural Network (CNN) trained on the EMNIST dataset. 

[![Gradio](https://img.shields.io/badge/Gradio-App-blue?logo=gradio)](https://gradio.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
---
## Demo
Try It Live Here:
[Draw a Digit ↗](https://huggingface.co/spaces/rezaenayati/Live_Digit_Recognition_Using_CNN)
---
## Overview
In this project I built a handwritten digit recognition using both a Dense Neural Network and CNNs. I tested with different amount of layers, units, learning rates and epochs. 
The EMNIST dataset come in 28 x 28 grayscale matrices, where each value represents the intesinty for that pixel. For the dense layers I had to unroll the matricies meaning that I had to reshape the matrix to make a vector of size 784. I then normalized the pixel values to [0,1] range by dividing by 255.0. This preprocessing step improves numerical stability and helps the model converge more efficiently during gradient descent and to make sure the gradients stay well scaled during back prop.  
