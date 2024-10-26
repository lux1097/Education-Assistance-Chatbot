import numpy as np
from transformers.image_utils import load_image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

device = 0 if torch.cuda.is_available() else -1
model_name = "vit-affectnet-custom-emotion-recognition"

# Load the image processor from a pre-trained model
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Load the image classification model from a pre-trained model and move it to the specified device (e.g., GPU or CPU)
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

# Define a function to apply the sigmoid activation function to the model's outputs
def sigmoid(_outputs):
    # Compute the sigmoid function: 1 / (1 + e^(-x))
    return 1.0 / (1.0 + np.exp(-_outputs))

# Define a function to apply the softmax activation function to the model's outputs
def softmax(_outputs):
    # Get the maximum value from the outputs to prevent overflow in the exponential calculation
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    # Subtract the max value, exponentiate the result, and compute the shifted exponential
    shifted_exp = np.exp(_outputs - maxes)
    # Divide by the sum of the shifted exponential values to normalize, obtaining the softmax probabilities
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

# Define a function to get the emotion label and score from an image
def get_emotion(image):
    # Process the image and obtain model inputs, converting them to PyTorch tensors
    model_inputs = image_processor(images=image, return_tensors='pt')
    # Move the model inputs to the same device as the model
    model_inputs = model_inputs.to(device)

    # Perform a forward pass through the model with the input data
    model_outputs = model(**model_inputs)

    # Retrieve the logits (raw model outputs), move them to CPU, detach from the computation graph, and convert to a NumPy array
    outputs = model_outputs["logits"][0].to(device).cpu().detach().numpy()

    # Compute softmax probabilities from the logits to obtain scores
    scores = softmax(outputs)

    # Create a list of dictionaries containing emotion labels and their corresponding scores
    dict_scores = [
        {"label": model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
    ]
    # Sort the list in descending order based on the scores
    dict_scores.sort(key=lambda x: x["score"], reverse=True)

    # Get the label with the highest score as the predicted emotion
    label = dict_scores[0]['label']
    # Get the highest score associated with the predicted label
    score = dict_scores[0]['score']

    # Return the predicted emotion label and its score
    return label, score
