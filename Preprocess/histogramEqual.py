import os
import cv2

def enhance_image(img_path):
    image = cv2.imread(img_path)

    # Convert the image to LAB color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the L channel (luminance)
    l_channel = lab_img[:,:,0]

    # Apply brightness preserving adaptive histogram equalization (BPAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l_channel)
    lab_img[:,:,0] = clahe_img

    # Convert the image back to RGB color space
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    return enhanced_img

if __name__ == "__main__":
    input_folder = "dataset/Sooty Mould"
    output_folder = "dataset with preprocess/preprocessed/Sooty Mould"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): 
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            enhanced_img = enhance_image(input_image_path)
            cv2.imwrite(output_image_path, enhanced_img)

    print("Done.")
