from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('skin_cancer_model_v2.keras')

# Function to predict the risk of malignant skin cancer
def predict_skin_cancer(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    risk_percentage = prediction[0][0] * 100

    return risk_percentage

# Upload an image
image_path = 'C:\\Users\\sreec\\OneDrive\\Desktop\\Dataset\\melanoma_cancer_dataset\\test\\benign\\melanoma_9605.jpg'
risk_percentage = predict_skin_cancer(image_path)

# Display the risk percentage
print(f'The risk of this skin being malignant is {risk_percentage:.2f}%')

# To visualize the uploaded image
plt.figure(figsize=(10, 5))

# Plot the image
plt.subplot(1, 2, 1)
img = image.load_img(image_path, target_size=(150, 150))
plt.imshow(img)
plt.title('Uploaded Image')
plt.axis('off')

# Plot the risk percentage
plt.subplot(1, 2, 2)
labels = ['Risk', 'No Risk']
sizes = [risk_percentage, 100 - risk_percentage]
colors = ['red', 'green']
explode = (0.1, 0)  # explode the 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Risk Assessment')

plt.show()