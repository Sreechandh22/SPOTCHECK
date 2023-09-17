# Existing import statements
import cv2
import matplotlib.pyplot as plt

def generate_synthetic_images(num_images):
    # Generate 'num_images' synthetic images
    print(f"Generating {num_images} synthetic images.")
    return ["image1", "image2"]  # Replace with actual image data

# Load the image from the specified path
synthetic_image = cv2.imread("C:\\Users\\sreec\\OneDrive\\Desktop\\Dataset\\melanoma_cancer_dataset\\test\\benign\\melanoma_9605.jpg")

# Convert the image from BGR to RGB (OpenCV loads images in BGR)
synthetic_image_rgb = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(synthetic_image_rgb)
plt.title("Synthetic Image")
plt.show()
