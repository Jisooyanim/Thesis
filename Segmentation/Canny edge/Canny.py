import os
import cv2

def cannyEdge(input_folder, output_folder, t_lower=100, t_upper=200, aperture_size=5, L2Gradient=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Process only image files
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            # Apply the Canny edge detection algorithm
            edge = cv2.Canny(img, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient)
            
            # Save the processed image in the output folder
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_edge.jpg")
            cv2.imwrite(output_path, edge)
            print(f"Processed: {filename} -> {output_path}")

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    cannyEdge(input_folder, output_folder)
