import os
import cv2
import numpy as np

def kmeans_segmentation(input_folder, output_folder, k=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(os.path.join(input_folder, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Reshape image pixel values
            pixel_vals = image.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to 8-bit integer values
            centers = np.uint8(centers)

            # Map labels to center points (RGB values)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((image.shape))

            output_file = os.path.join(output_folder, file)
            cv2.imwrite(output_file, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    print('Done!')

if __name__ == '__main__':
    input_folder = ''
    output_folder = ''
    kmeans_segmentation(input_folder, output_folder)