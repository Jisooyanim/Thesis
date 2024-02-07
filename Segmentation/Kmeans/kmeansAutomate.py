import os
import cv2
import numpy as np

def kmeans_segmentation(input_folder, output_folder, k=3):
    # Ensure the output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file
    for file in files:
        # Check if the file is an image
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image = cv2.imread(os.path.join(input_folder, file))

            # Convert image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reshape image pixel values
            pixel_vals = image.reshape((-1, 3))

            # Convert pixel values to float32
            pixel_vals = np.float32(pixel_vals)

            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

            # Apply k-means clustering
            _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to 8-bit integer values
            centers = np.uint8(centers)

            # Map labels to center points (RGB values)
            segmented_data = centers[labels.flatten()]

            # Reshape data into the original image dimensions
            segmented_image = segmented_data.reshape((image.shape))

            # Save segmented image
            output_file = os.path.join(output_folder, file)
            cv2.imwrite(output_file, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
            print(f"Segmented image saved: {output_file}")

if __name__ == '__main__':
    input_folder = 'Images/bgremoved'
    output_folder = 'Images/Without preprocess/Kmeans'
    kmeans_segmentation(input_folder, output_folder)