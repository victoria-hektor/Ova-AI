# Ova-AI
This is the code for my thesis title "Integration of a Weighted Ensemble of ResNet50 and InceptionV3 Models for Enhanced PCOS Detection Using Ultrasound Images"

OvaAI is a clinical decision support system designed to assist in the detection of Polycystic Ovary Syndrome (PCOS) using deep learning models. This project combines a weighted ensemble of pre-trained models (ResNet50 and InceptionV3) with an intuitive web interface for data upload and prediction.

Features:
Ensemble model using ResNet50 and InceptionV3 with weighted averaging.
Data augmentation using GAN-enhanced techniques.
Web dashboard for seamless image upload and classification.
Modular design for scalability and integration.

Flow of the Project:

1. Model Training
Train individual models (ResNet50 and InceptionV3) on your dataset using the scripts provided.
These models will generate .h5 files for storage. Due to GitHub's policies, pre-trained individual models are not included in this repository.
You must train the models on your dataset by following the instructions in the training/ directory.

3. Model Inference
An ensemble model (already included: ensemble_model.h5) combines predictions from the individual models to enhance classification accuracy.

5. Web Application
The web interface is based on the Soft UI Dashboard (see license details below) and provides a user-friendly experience for uploading medical images and receiving predictions.

Installation and Setup:
Prerequisites - Python 3.8 or above
Required libraries: TensorFlow, Keras, Flask, and other dependencies listed in requirements.txt.

Steps to Clone and Use
Clone the repository:

git clone https://github.com/yourusername/ovaai.git
cd ovaai

Install the required dependencies:

pip install -r requirements.txt

Train the individual models (if not already trained):

Navigate to the training/ folder and execute the training scripts:

python train_resnet50.py
python train_inceptionv3.py

Note: The individual .h5 files for ResNet50 and InceptionV3 must be placed in the models/ directory after training.

Run the Flask web application:

python app.py
Access the dashboard at http://127.0.0.1:5000.

Directory Structure:

OvaAI/
├── app.py                  # Main Flask application
├── models/
│   ├── ensemble_model.h5   # Pre-trained ensemble model
│   ├── resnet50.h5         # Placeholder for ResNet50 model (train locally)
│   ├── inceptionv3.h5      # Placeholder for InceptionV3 model (train locally)
├── static/                 # Static files for the web interface
├── templates/              # HTML templates for the web interface
├── training/               # Training scripts for individual models
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file for the dashboard
└── README.md               # Project documentation
Usage Notes

Pre-trained Models:
Only the ensemble_model.h5 file is provided in this repository to streamline predictions.
Users must train and add the individual models (resnet50.h5, inceptionv3.h5) to the models/ directory.

Data:
Use your dataset to train the models. For data preparation, refer to the flowcharts included in the repository (flowcharts/ folder).

Dashboard Licensing:
The dashboard interface is based on the Soft UI Dashboard by Creative Tim, which is licensed for personal use. Ensure compliance with the license agreement detailed in the LICENSE file.

Flowcharts:
Refer to the flowcharts/ directory for detailed visual representations of the system's workflows:

Ensemble Model Workflow
Web Application Workflow
Individual Model Training Workflow
PCOS Prediction Flow

Future Work:
Integration with additional medical imaging datasets.
Development of a more comprehensive GAN model for data augmentation.
Extending the dashboard for multi-class classification tasks.
