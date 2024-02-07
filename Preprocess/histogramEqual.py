import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_image(img_path):
    # Load the image
    original_img = cv2.imread(img_path)

    # Convert the image to LAB color space
    lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)

    # Extract the L channel (luminance)
    l_channel = lab_img[:,:,0]

    # Apply brightness preserving adaptive histogram equalization (BPAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l_channel)

    # Replace the original luminance channel with the enhanced channel
    lab_img[:,:,0] = clahe_img

    # Convert the image back to RGB color space
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    # Plotting the original and enhanced images
    plt.figure(figsize=(10, 5))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Plot enhanced image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced Image")
    plt.axis('off')

    plt.show()

    # Save the enhanced image
    cv2.imwrite("enhanced_image.jpg", enhanced_img)

if __name__ == "__main__":
    # Provide the path to the input image
    image_path = "test_images/20211008_124249 (Custom).jpg"

    # Call the function to enhance the image
    enhance_image(image_path)
