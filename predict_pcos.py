import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the individual models
inception_model = load_model('inceptionv3_model.h5')
resnet_model = load_model('resnet50_model.h5')

# Function to preprocess and predict the image
def prepare_image(file_path):
    """
    Preprocess the image to the required input format for the model.
    Args:
    file_path (str): Path to the image file.

    Returns:
    np.array: Preprocessed image ready for prediction.
    """
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale as per the training process
    return img_array

def predict_image(file_path):
    """
    Predict the class of the image using the ensemble of Inception and ResNet models.
    Args:
    file_path (str): Path to the image file.

    Returns:
    int: Predicted class (0 or 1).
    """
    # Pre-process the image
    img_array = prepare_image(file_path)
    
    # Predict using individual models
    y_pred_inception = inception_model.predict(img_array)
    y_pred_resnet = resnet_model.predict(img_array)
    
    # Assign weights to each model's prediction
    weight_inception = 0.7
    weight_resnet = 0.3

    # Calculate the weighted average of predictions
    y_pred_ensemble = (weight_inception * y_pred_inception) + (weight_resnet * y_pred_resnet)

    print(f"Raw ensemble prediction: {y_pred_ensemble[0]}")

    # Convert to binary class label (0 or 1)
    predicted_class = int(y_pred_ensemble[0] > 0.5)
    
    return predicted_class