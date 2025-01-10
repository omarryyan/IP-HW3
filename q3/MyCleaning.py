# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image (this is the image we want to work with)
image_path = "broken.jpg"  # Replace with the path to your actual image
original_image = cv2.imread(image_path)

# Check if the image is loaded successfully
if original_image is None:
    raise FileNotFoundError("Image file not found. Ensure it is in the correct path.")

#------------------------------------Part a ---------------------------------

# Apply Median Blur for salt-and-pepper noise
median_image = cv2.medianBlur(original_image, 3)

# Make sure the image is gray scale
median_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY)

# Apply the bilateral filter to the median image
our_fixed_image=cv2.bilateralFilter(median_image, 5, 50, 20)


cv2.imshow("our fixed image",our_fixed_image)
cv2.imwrite("OurFixedImage.jpg",our_fixed_image)
cv2.waitKey(0)

#-----------------------------------------Part b ------------------------------

# Load the noisy images dataset
data = np.load('noised_images (1).npy')  # Assuming the data is loaded properly

# Average all noisy images to reduce noise
average_image = np.mean(data, axis=0).astype(np.uint8)

cv2.imshow("average image",average_image)
cv2.imwrite("AverageImage.jpg",average_image)
cv2.waitKey(0)

# Visualize the images
plt.figure(figsize=(10, 5))
# Original image
plt.subplot(131)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis("off")

# Denoised Image (after bilateral filtering)
plt.subplot(132)
plt.title("Our fixed Image")
plt.imshow(our_fixed_image, cmap='gray')
plt.axis("off")

# Average Noised Image
plt.subplot(133)
plt.title("Average Noised Image using .npy folder")
plt.imshow(average_image, cmap='gray')
plt.axis("off")

plt.show()
