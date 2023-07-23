from flask import Flask, render_template,request
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import base64


app = Flask(__name__)
genuine_images_path = "data/genuine"
forged_images_path = "data/forged"

genuine_image_filenames = os.listdir(genuine_images_path)
forged_image_filenames = os.listdir(forged_images_path)

genuine_image_features = []
forged_image_features = []

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Add a check to ensure the threshold_image is not None
    if threshold_image is None:
        raise ValueError("Threshold image is None.")
    
    # Add a check to ensure the threshold_image is not empty
    if threshold_image.size == 0:
        raise ValueError("Threshold image is empty.")

    resized_image = cv2.resize(threshold_image, (200, 100)).astype(np.uint8)
    return resized_image.flatten()




for name in genuine_image_filenames:
    image_path = genuine_images_path + "/" + name
    feature_vector = preprocess_image(image_path)
    genuine_image_features.append(feature_vector)

for name in forged_image_filenames:
    image_path = forged_images_path + "/" + name
    feature_vector = preprocess_image(image_path)
    forged_image_features.append(feature_vector)

# Create labels for the images (1 for genuine, 0 for forged)
genuine_labels = np.ones(len(genuine_image_features))
forged_labels = np.zeros(len(forged_image_features))

# Combine genuine and forged features and labels
all_features = genuine_image_features + forged_image_features
all_labels = np.concatenate((genuine_labels, forged_labels))

# Normalize the feature vectors
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, all_labels, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the trained model and scaler
with open("model.pkl", "wb") as file:
    pickle.dump(svm_model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)



@app.route('/')
def home():
    return render_template('index.html')

with open("model.pkl", "rb") as file:
    svm_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)



@app.route('/verify', methods=['POST'])
def verify():
    # Get the selected image file from the form
    image_file = request.files['image']

    # Save the image file to a temporary location
    image_path = 'temp.jpg'
    image_file.save(image_path)

    # Preprocess the image
    feature_vector = preprocess_image(image_path)
    scaled_feature = scaler.transform(feature_vector.reshape(1, -1))

    # Make prediction
    prediction = svm_model.predict(scaled_feature)

    # Remove the temporary image file
    os.remove(image_path)

    # Return the prediction result
    if prediction == 1:
        result = 'The signature is genuine.'
    else:
        result = 'The signature is forged.'
    return render_template('result.html', result=result)
     # Base64 encode the preprocessed image
    # Preprocess the image
    ''' preprocessed_image = preprocess_image(image_path)
    _, encoded_image = cv2.imencode('.png', preprocessed_image)
    preprocessed_image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')'''
    return render_template('result.html', result=result)

    # Pass the result and preprocessed image URL to the template
    #return render_template('result.html', result=result, preprocessed_image_url=f"data:image/png;base64,{preprocessed_image_base64}")







if __name__ == '__main__':
    app.run(debug=True)


