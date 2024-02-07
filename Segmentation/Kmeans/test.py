import numpy as np
import cv2

def kmeans_image_segmentation(image, k):
    # Convert image to float32
    img = np.float32(image.reshape((-1, 3)))

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape the result back to the original image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

# Load the image
image = cv2.imread('preprocessed/20211008_124249 (Custom).jpg')

# Number of clusters (K) can be determined at runtime based on the diseased leaf image
K = 3

# Perform K-means image segmentation
segmented_image = kmeans_image_segmentation(image, K)

# Save the segmented image
cv2.imwrite('segmented_image.jpg', segmented_image)

print("Segmented image saved as 'segmented_image.jpg'")
