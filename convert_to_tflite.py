import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("C:\\Users\\sreec\\OneDrive\\Desktop\\skin_cancer_model_v2.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("C:\\Users\\sreec\\OneDrive\\Desktop\\skin_cancer_model_v2.tflite", "wb") as f:
    f.write(tflite_model)
