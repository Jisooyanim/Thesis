# Loading required libraries
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

image = cv2.imread('dataset with preprocess/preprocessed/Bacterial Canker/IMG_20211106_120955 (Custom).jpg') 

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
pixel_vals = image.reshape((-1,3)) # numpy reshape operation -1 unspecified 

# Convert to float type only for supporting cv2.kmean
pixel_vals = np.float32(pixel_vals)

#criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

# Choosing number of cluster
k = 5

retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

# convert data into 8-bit values 
centers = np.uint8(centers) 

segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)

# reshape data into the original image dimensions 
segmented_image = segmented_data.reshape((image.shape)) 

plt.figure(figsize=(10, 5))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Plot the processed image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Processed Image')
plt.axis('off')

plt.show()
