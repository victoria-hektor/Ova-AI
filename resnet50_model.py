import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Define paths to your data
train_dir = r'C:\Users\Student\Project_PCOS\PCOS_Detection\data\train'
test_dir = r'C:\Users\Student\Project_PCOS\PCOS_Detection\data\test'

# ImageDataGenerator for preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced rotation
    width_shift_range=0.2,  # Reduced width shift
    height_shift_range=0.2,  # Reduced height shift
    shear_range=0.2,  # Reduced shear
    zoom_range=0.2,  # Reduced zoom
    horizontal_flip=True,  # Kept horizontal flip
    vertical_flip=False,  # Removed vertical flip
    brightness_range=[0.8, 1.2],  # Less aggressive brightness adjustment
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained ResNet50 model
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune more layers
for layer in resnet_base.layers[-10:]:
    layer.trainable = True

# Add custom layers
x = resnet_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Slightly less regularisation
x = Dropout(0.5)(x)  # Slightly reduced dropout
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=resnet_base.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model with k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
fold_no = 1

for train_index, val_index in kf.split(np.arange(train_generator.samples)):
    print(f"Training fold {fold_no}...")
    train_generator.reset()
    validation_generator.reset()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )

    # Save the training history for this fold
    with open(f'resnet50_history_fold_{fold_no}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    fold_no += 1

# Save the trained model
model.save('resnet50_model_2.h5')

# Evaluate the model on validation data
validation_generator.reset()
y_true = validation_generator.classes
y_pred = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

# Calculate and print metrics
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)
print("\nClassification Report:")
class_report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys())
print(class_report)
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Visualise performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()