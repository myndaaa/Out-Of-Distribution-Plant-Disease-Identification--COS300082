# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from shutil import copyfile

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
SIMILAR_CROPS_FOLDER = 'static/similar_crops/'
SIMILAR_DISEASES_FOLDER = 'static/similar_diseases/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Ensure upload and similar images directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIMILAR_CROPS_FOLDER, exist_ok=True)
os.makedirs(SIMILAR_DISEASES_FOLDER, exist_ok=True)

# Load models
crop_model_path = os.path.join('models', 'crop.h5')
disease_model_path = os.path.join('models', 'disease.h5')

try:
    crop_model = load_model(crop_model_path)
    disease_model = load_model(disease_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    crop_model = None
    disease_model = None

# Load training data
train_csv_path = os.path.join('datasets', 'PV_train.csv')
if os.path.exists(train_csv_path):
    train_data = pd.read_csv(train_csv_path, header=None, names=["image_name", "crop_class", "disease_class"])
    # Convert class columns to strings
    train_data["crop_class"] = train_data["crop_class"].astype(str)
    train_data["disease_class"] = train_data["disease_class"].astype(str)
else:
    train_data = pd.DataFrame(columns=["image_name", "crop_class", "disease_class"])
    print(f"Training CSV not found at {train_csv_path}")

# Define image directories
plant_vil_dir = os.path.join('datasets', 'plantvillage')
plant_doc_dir = os.path.join('datasets', 'plantdoc')

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_flask(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Preprocess as per your model's training
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(image_lab, cv2.COLOR_Lab2RGB)
    enhanced_image_filtered = cv2.medianBlur(enhanced_image, 3)
    # Resize to 224x224
    image_resized = cv2.resize(enhanced_image_filtered, (224, 224))
    # Convert to array and rescale
    image_array = img_to_array(image_resized) / 255.0
    # Expand dimensions to match model input
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_similar_images(class_type, class_label, num_images=3):
    """
    Fetches similar images from the training dataset based on the predicted class.
    """
    if class_type == 'crop':
        df = train_data[train_data['crop_class'] == class_label]
        image_dir = plant_vil_dir
    elif class_type == 'disease':
        df = train_data[train_data['disease_class'] == class_label]
        image_dir = plant_vil_dir  
    else:
        return []
    
    # Check if dataframe is not empty
    if df.empty:
        return []
    
    # Randomly select images
    selected = df.sample(n=min(num_images, len(df)), random_state=np.random.randint(0, 10000))
    
    image_paths = []
    for _, row in selected.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
    
    # Define the folder to store similar images
    similar_folder = SIMILAR_CROPS_FOLDER if class_type == 'crop' else SIMILAR_DISEASES_FOLDER
    # Clear previous similar images
    for f in os.listdir(similar_folder):
        file_path = os.path.join(similar_folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    similar_image_urls = []
    for image_path in image_paths:
        filename = secure_filename(os.path.basename(image_path))
        dest_path = os.path.join(similar_folder, filename)
        if not os.path.exists(dest_path):
            # Copy image to the similar images folder
            copyfile(image_path, dest_path)
        # Construct the correct URL path
        if class_type == 'crop':
            url_path = f'similar_crops/{filename}'
        else:
            url_path = f'similar_diseases/{filename}'
        similar_image_urls.append(url_for('static', filename=url_path))
    
    return similar_image_urls

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if it's an upload request
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(upload_path)
                except Exception as e:
                    flash(f"Error saving file: {e}")
                    return render_template('index.html')
                
                # Clear similar images folders
                for folder in [SIMILAR_CROPS_FOLDER, SIMILAR_DISEASES_FOLDER]:
                    for f in os.listdir(folder):
                        file_path = os.path.join(folder, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                flash("Image uploaded successfully.")
                return render_template('index.html', uploaded_image=filename)
            else:
                flash("Invalid file type. Please upload an image.")
                return render_template('index.html')
        
        # Check if it's a classify request
        elif 'classify' in request.form:
            filename = request.form.get('uploaded_image', '')
            if filename == '':
                flash("No image uploaded.")
                return render_template('index.html', uploaded_image=None)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(upload_path):
                flash("Uploaded image not found.")
                return render_template('index.html', uploaded_image=None)
            # Preprocess image
            image_array = preprocess_image_flask(upload_path)
            if image_array is None:
                flash("Invalid image uploaded.")
                return render_template('index.html', uploaded_image=filename)
            # Predict crop class
            if crop_model:
                crop_pred = crop_model.predict(image_array)
                crop_class_idx = np.argmax(crop_pred, axis=1)[0]
                # Assuming you have a mapping from index to class label
                crop_class_label = str(crop_class_idx)  # Replace with actual mapping
            else:
                crop_class_label = "Model not loaded"
            
            # Predict disease class
            if disease_model:
                disease_pred = disease_model.predict(image_array)
                disease_class_idx = np.argmax(disease_pred, axis=1)[0]
                # Assuming you have a mapping from index to class label
                disease_class_label = str(disease_class_idx)  # Replace with actual mapping
            else:
                disease_class_label = "Model not loaded"
            
            # Get similar images
            similar_crops = get_similar_images('crop', crop_class_label)
            similar_diseases = get_similar_images('disease', disease_class_label)
            
            return render_template('index.html',
                                   uploaded_image=filename,
                                   crop_class=crop_class_label,
                                   disease_class=disease_class_label,
                                   similar_crops=similar_crops,
                                   similar_diseases=similar_diseases)
        
        # Check if it's a reset request
        elif 'reset' in request.form:
            filename = request.form.get('uploaded_image', '')
            if filename:
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(upload_path):
                    try:
                        os.remove(upload_path)
                    except Exception as e:
                        flash(f"Error deleting file: {e}")
            # Remove similar images
            for folder in [SIMILAR_CROPS_FOLDER, SIMILAR_DISEASES_FOLDER]:
                for f in os.listdir(folder):
                    file_path = os.path.join(folder, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            flash("Reset successfully.")
            return render_template('index.html', uploaded_image=None)
        
        else:
            flash("Invalid request.")
            return render_template('index.html')
    
    else:
        return render_template('index.html')

# Routes for pages
@app.route('/archi')
def archi():
    return render_template('archi.html')

@app.route('/contributors')
def contributors():
    return render_template('contributors.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
