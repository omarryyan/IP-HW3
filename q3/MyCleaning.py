import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy images dataset
data = np.load('noised_images (1).npy')

# Assuming that 'data' contains 100 noisy images
# Average all noisy images to reduce noise
average_image = np.mean(data, axis=0).astype(np.uint8)

# Read the original image
image_path = "broken.jpg"
image = cv2.imread(image_path)
cv2.imshow("Original", image)

# Convert the image to grayscale (if needed)
fixed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Median Blur for salt-and-pepper noise
fixed_image = cv2.medianBlur(fixed_image, 3)

# Apply Bilateral Filter to reduce noise while keeping edges sharp
fixed_image = cv2.bilateralFilter(fixed_image, d=4, sigmaColor=50, sigmaSpace=50)

# Apply Gaussian Blur to smooth the image
fixed_image = cv2.GaussianBlur(fixed_image, (3, 3), 0)

# Show the result after denoising
cv2.imshow("Denoised Image", fixed_image)

# Additionally, show the average of noisy images
cv2.imshow("Average Noised Image", average_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
