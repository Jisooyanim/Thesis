#test2
import os
import cv2
import numpy as np

def otsu(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    for file in files:
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, thresholded_image)

        print(f"Processed: {file}")

input_folder = ''
output_folder = ''

otsu(input_folder, output_folder)