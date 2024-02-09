import os
import shutil
import random

# Path to your main dataset folder
dataset_folder = "dataset"
# Path to save the split dataset
output_folder = "dataset maskrcnn"

# Ratio of images for training and validation
train_ratio = 0.8
valid_ratio = 1 - train_ratio

# Iterate through each subfolder (class folder)
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    if os.path.isdir(class_path):
        # Create train and valid folders for the class in the output folder
        train_class_folder = os.path.join(output_folder, "train", class_folder)
        valid_class_folder = os.path.join(output_folder, "valid", class_folder)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(valid_class_folder, exist_ok=True)
        
        # List all images in the class folder
        images = os.listdir(class_path)
        
        # Calculate the number of images for training and validation
        num_train = int(len(images) * train_ratio)
        num_valid = len(images) - num_train
        
        # Randomly shuffle the images
        random.shuffle(images)
        
        # Move images to train and valid folders
        for i, image in enumerate(images):
            src = os.path.join(class_path, image)
            if i < num_train:
                dst = os.path.join(train_class_folder, image)
            else:
                dst = os.path.join(valid_class_folder, image)
            shutil.copy(src, dst)
