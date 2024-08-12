import cv2
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Root directory containing subdirectories with images
root_directory = "../Piotr-NB-Dataset/"
# Number of images to randomly select from each subdirectory
n_images_per_subdirectory = 5

# List to store pixel values for all selected regions
all_pixel_values = []

# Iterate through each subdirectory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    if os.path.isdir(subdir_path):
        # Iterate through the second level of directories within each subdirectory
        for sub_subdir in os.listdir(subdir_path):
            sub_subdir_path = os.path.join(subdir_path, sub_subdir)
            if os.path.isdir(sub_subdir_path):
                # List all images in the current subdirectory
                images = [os.path.join(sub_subdir_path, filename) for filename in os.listdir(sub_subdir_path) if
                          filename.endswith(('.jpg', '.jpeg', '.png'))]

                # Randomly select N images from the list if available
                selected_images = random.sample(images, min(n_images_per_subdirectory, len(images)))

                # Iterate through selected images
                for image_path in selected_images:
                    # Load the image
                    image = cv2.imread(image_path)

                    # Select ROI manually
                    roi = cv2.selectROI(image)

                    cv2.destroyAllWindows()

                    # Crop the selected region from the image
                    roi_cropped = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

                    # Flatten the pixel values and store them in the list
                    all_pixel_values.extend(roi_cropped.reshape(-1))


# Save pixel values to a text file
np.savetxt('pixel_values.txt', all_pixel_values, fmt='%d')

# Calculate min and max pixel values
min_pixel_value = np.min(all_pixel_values)
max_pixel_value = np.max(all_pixel_values)

print("Min Pixel Value:", min_pixel_value)
print("Max Pixel Value:", max_pixel_value)

# Generate distribution plot
sns.histplot(all_pixel_values, kde=True)
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
