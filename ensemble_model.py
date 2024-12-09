import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(tf.keras.__version__)
print(tf.__version__)

# Load the saved models
inception_model = load_model('inceptionv3_model.h5')
resnet_model = load_model('resnet50_model.h5')

# Define paths to your data
test_dir = r'C:\Users\Student\Project_PCOS\PCOS_Detection\data\test'

# ImageDataGenerator for preprocessing and augmentation
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load the test data
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Obtain predictions from both models
y_pred_inception = inception_model.predict(validation_generator, steps=len(validation_generator), verbose=1)
y_pred_resnet = resnet_model.predict(validation_generator, steps=len(validation_generator), verbose=1)

# Assign weights based on performance
weight_inception = 0.7
weight_resnet = 0.3

# Calculate weighted average of predictions
y_pred_ensemble = (weight_inception * y_pred_inception) + (weight_resnet * y_pred_resnet)

# Convert to binary class labels
y_pred_classes_ensemble = (y_pred_ensemble > 0.5).astype("int32")

# Save the ensemble model predictions
np.save('ensemble_predictions.npy', y_pred_classes_ensemble)

# The weights are not directly applicable here but for saving purpose, we just save the structure.
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the ensemble model
ensemble_model.save('ensemble_model.h5')

# Obtain the true labels
y_true = validation_generator.classes

# Evaluate ensemble performance
accuracy_ensemble = accuracy_score(y_true, y_pred_classes_ensemble)
precision_ensemble = precision_score(y_true, y_pred_classes_ensemble)
recall_ensemble = recall_score(y_true, y_pred_classes_ensemble)
f1_score_ensemble = f1_score(y_true, y_pred_classes_ensemble)

print("Ensemble Accuracy:", accuracy_ensemble)
print("Ensemble Precision:", precision_ensemble)
print("Ensemble Recall:", recall_ensemble)
print("Ensemble F1 Score:", f1_score_ensemble)

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Ensemble Confusion Matrix')
plt.show()