import os
from rembg import remove
import cv2
import numpy as np
from PIL import Image

def remove_background(img_path):
    img = Image.open(img_path)
    output = remove(np.array(img)) 

    return output

def enhance_image(removed_bg_img):
    # Convert the image to LAB color space
    lab_img = cv2.cvtColor(removed_bg_img, cv2.COLOR_RGB2LAB)

    # Luminance channel
    l_channel = lab_img[:,:,0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l_channel)
    lab_img[:,:,0] = clahe_img

    # Convert the image back to RGB color space
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    # enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    return enhanced_img

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_img_path = os.path.join(input_folder, filename)
            removed_bg_img = remove_background(input_img_path)
            enhanced_img = enhance_image(removed_bg_img)
            output_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_img_path, enhanced_img)

if __name__ == '__main__':
    input_folder = 'test_images'
    output_folder = 'preprocessed'

    process_images_in_folder(input_folder, output_folder)