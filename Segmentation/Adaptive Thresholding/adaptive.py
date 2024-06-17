#test3
import os
import numpy as np
import time
from PIL import Image

input_folder = 'dataset with preprocess/preprocessed/Sooty Mould'
output_folder = 'dataset with preprocess/Segmented/Adaptive/Sooty Mould'
os.makedirs(output_folder, exist_ok=True)

def load_image(filepath):
    return Image.open(filepath)

def save_image(image, filepath):
    image = image.convert('L')
    image.save(filepath)

def image2array(image):
    return np.array(image)

def array2image(image_array):
    return Image.fromarray(image_array)

def rgb2grayscale(image_array):
    return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

def compute_integral_image(image_array, height, width):
    integral_image = np.empty_like(image_array)

    for w in range(width):
        sum_ = 0
        for h in range(height):
            sum_ += image_array[h][w]
            if w == 0:
                integral_image[h][w] = sum_
            else:
                integral_image[h][w] = sum_ + integral_image[h][w - 1]
    return integral_image

def adaptive_thresholding(image_array, height, width, s, t):
    integral_image = compute_integral_image(image_array, height, width)
    output_array = np.empty_like(image_array)

    for w in range(width):
        for h in range(height):
            x1 = w - s
            if x1 <= 0:
                x1 = 1
            y1 = h - s
            if y1 <= 0:
                y1 = 1
            x2 = w + s
            if x2 >= width:
                x2 = width - 1
            y2 = h + s
            if y2 >= height:
                y2 = height - 1
            count = (x2 - x1) * (y2 - y1)
            sum_ = integral_image[y2][x2] - integral_image[y1 - 1][x2] - integral_image[y2][x1 - 1] + integral_image[y1 - 1][x1 - 1]
            if image_array[h][w] * count <= sum_ * ((100 - t) / 100):
                output_array[h][w] = 0
            else:
                output_array[h][w] = 255
    return output_array

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, filename)

        input_image = load_image(input_filepath)
        image_array = rgb2grayscale(image2array(input_image))

        height = image_array.shape[0]
        width = image_array.shape[1]
        s = int(width / 16)
        t = 15

        start = time.time()
        output_array = adaptive_thresholding(image_array, height, width, s, t)
        end = time.time()

        print('Processing', filename)
        # print('Execution time:', end - start)

        output_image = array2image(output_array)
        save_image(output_image, output_filepath)
