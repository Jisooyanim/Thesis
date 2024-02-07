from rembg import remove 
from PIL import Image 
import matplotlib.pyplot as plt
import os

# Load the original image
# img = Image.open('test_images/20211008_124256 (Custom).jpg')

# # Process the image
# output = remove(img)
# output.save('processed_image.png')

# # Plot the original and processed image side by side
# plt.figure(figsize=(10, 5))

# # Plot the original image
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.axis('off')

# # Plot the processed image
# plt.subplot(1, 2, 2)
# plt.imshow(output)
# plt.title('Processed Image')
# plt.axis('off')

# plt.show()

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Process each image
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            # Load the original image
            img_path = os.path.join(input_folder, file)
            img = Image.open(img_path)

            # Process the image
            output = remove(img)

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, file.split('.')[0] + '_processed.png')
            output.save(output_path)

            print(f"Processed {file} and saved as {output_path}")

# Specify input and output folders
input_folder = 'dataset/Bacterial Canker'
output_folder = 'dataset without preprocess/Bacterial Canker'

# Process images in the input folder and save them to the output folder
process_images(input_folder, output_folder)