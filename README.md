SKIN CANCER DETECTION APP

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Create uploads folder if not exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the pre-trained model
try:
    model = load_model('skin_cancer_model_v2.keras')
except:
    print("Could not load model. Make sure the model exists.")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '' and uploaded_file.filename.endswith(('.jpg', '.png')):
            image_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(image_path)
            risk_percentage = predict_skin_cancer(image_path)
            return render_template('result.html', risk_percentage=risk_percentage, image_path=image_path)
    return render_template('index.html')

def predict_skin_cancer(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    risk_percentage = prediction[0][0] * 100

    return risk_percentage

if __name__ == '__main__':
    app.run(debug=True)
