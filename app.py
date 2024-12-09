import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from predict_pcos import predict_image

app = Flask(__name__, template_folder="OvaAI_Dashboard/pages", static_folder="OvaAI_Dashboard/assets")

# Setting a secret key for session management and flashing messages
app.secret_key = 'supersecretkey'

@app.route('/')
def index():
    return render_template('dashboard.html')

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predict_pcos():
    if request.method == 'POST':
        print("POST request received")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(f"File received: {file.filename}")
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)  # Use the updated ensemble function
            print(f"Prediction received: {prediction}")
            classification = "Negative" if prediction == 1 else "Positive"
            print(f"Classification: {classification}")
            return render_template('dashboard.html', classification=classification)
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)