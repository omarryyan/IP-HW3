# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy images dataset (this is assuming you have noisy images to work with)
data = np.load('noised_images (1).npy')  # Assuming the data is loaded properly

# Average all noisy images to reduce noise
average_image = np.mean(data, axis=0).astype(np.uint8)

# Read the original image (this is the image you want to work with)
image_path = "broken.jpg"  # Replace with the path to your actual image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError("Image file not found. Ensure it is in the correct path.")

# Convert the image to grayscale (if needed)
fixed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Median Blur for salt-and-pepper noise
fixed_image = cv2.medianBlur(fixed_image, 3)

def gaussian(x, sigma):
    """Gaussian function."""
    return np.exp(-0.5 * (x / sigma) ** 2)

# Parameters for the bilateral filter
radius = 5
stdSpatial = 100
stdIntensity = 20

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    """
    Denoises an image using a bilateral filter.

    :param im: 2D numpy array representing the grayscale image.
    :param radius: Radius of the neighborhood window (integer).
    :param stdSpatial: Standard deviation for spatial Gaussian.
    :param stdIntensity: Standard deviation for intensity Gaussian.
    :return: Denoised image as a 2D numpy array.
    """
    im = im.astype(np.float64)  # Ensure working with float64 for accuracy
    rows, cols = im.shape

    # Define the window size
    window_size = 2 * radius + 1

    # Pad the image to handle edges (padding with reflect mode)
    padded_im = np.pad(im, radius, mode='reflect')

    # Create spatial Gaussian kernel
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = gaussian(np.sqrt(x**2 + y**2), stdSpatial)

    # Reshape the spatial Gaussian into a 2D array (window_size x window_size)
    gs = gs / gs.sum()  # Normalize spatial Gaussian (this ensures no change in image brightness)

    # Create a shifted view of the image for easy access to neighborhoods
    windows = np.lib.stride_tricks.sliding_window_view(padded_im, (window_size, window_size))

    # Reshape windows for broadcasting
    windows = windows.reshape((rows, cols, window_size, window_size))

    # Calculate the intensity differences for each pixel in the window
    intensity_diff = windows - im[:, :, np.newaxis, np.newaxis]

    # Compute intensity Gaussian (gi) for the entire image
    gi = gaussian(intensity_diff, stdIntensity)

    # Apply the spatial and intensity Gaussian together
    weights = gs * gi

    # Normalize the combined weight matrix
    weights = weights / weights.sum(axis=(2, 3), keepdims=True)

    # Multiply the window values with the weights and sum them
    filtered_im = np.sum(weights * windows, axis=(2, 3))

    # Convert the result back to uint8
    return np.clip(filtered_im, 0, 255).astype(np.uint8)

# Apply the bilateral filter to the grayscale image
filtered_image = clean_Gaussian_noise_bilateral(fixed_image, radius, stdSpatial, stdIntensity)

# Apply Gaussian Blur to smooth the image (if needed)
fixed_image = cv2.GaussianBlur(fixed_image, (3, 3), 0)

# Visualize the images
plt.figure(figsize=(10, 5))
cv2.imshow('fixed image', fixed_image)
cv2.waitKey(0)
# Original image
plt.subplot(131)
plt.title("Original Image")
plt.imshow(fixed_image, cmap='gray')
plt.axis("off")

# Denoised Image (after bilateral filtering)
plt.subplot(132)
plt.title("Denoised Image")
plt.imshow(filtered_image, cmap='gray')
plt.axis("off")

# Average Noised Image
plt.subplot(133)
plt.title("Average Noised Image")
plt.imshow(average_image, cmap='gray')
plt.axis("off")

plt.show()
