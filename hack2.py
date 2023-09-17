from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Initialize VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the layers

# Build the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Initialize ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
# Load data from directories
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\sreec\OneDrive\Desktop\Dataset\melanoma_cancer_dataset\train',  # Using a raw string
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\sreec\OneDrive\Desktop\Dataset\melanoma_cancer_dataset\test',  # Using a raw string
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy', 'Precision', 'Recall']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=20,
    validation_data=test_generator,
    validation_steps=20
)

# Save the model
model.save('improved_skin_cancer_model.keras')

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Precision: {test_precision * 100:.2f}%')
print(f'Test Recall: {test_recall * 100:.2f}%')

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()