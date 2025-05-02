import gradio as gr
import numpy as np 
import tensorflow as tf

# Load the model 
model = tf.keras.models.load_model("cnn_model_v2.h5")


def predict(input_dict):
    if not input_dict or "composite" not in input_dict:
        return {str(i): 0.0 for i in range(10)}

    image = input_dict["composite"]

    # Remove alpha channel (RGBA → RGB)
    image = image[:, :, :3]

    # Convert to grayscale
    image = tf.image.rgb_to_grayscale(image)

    # Resize to 28x28
    image = tf.image.resize(image, [28, 28])

    # Invert colors and normalize
    image = 255 - image
    image = image / 255.0
    image = image.numpy().reshape(-1, 28, 28, 1)

    # Predict
    prediction = model.predict(image)
    probs = tf.nn.softmax(prediction[0]).numpy()

    if probs.max() < 0.5:
        return "Please Try Again."
    else:
        return {str(i): float(probs[i]) for i in range(10)}

# Create the Gradio interface with appropriate settings
sketchpad = gr.Sketchpad()  # <- clean and version-safe
label = gr.Label(num_top_classes=3)

gr.Interface(
    fn=predict,
    inputs=sketchpad,
    outputs=label,
    title="MNIST Digit Sketch Pad",
    description="Draw a digit (0–9) and this model will try to guess what number you wrote! It’s trained on the original MNIST dataset of handwritten digits.",
).launch(debug = False
