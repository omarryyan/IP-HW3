import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma):
    """Gaussian function."""
    return np.exp(-0.5 * (x / sigma) ** 2)

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

# Example Usage
original_image_path = 'balls.jpg'  # Replace with your actual file name
color_image = cv2.imread(original_image_path)  # Load color image

if color_image is None:
    raise FileNotFoundError("Image file not found. Ensure it is in the same folder as this script.")

# Convert the color image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Parameters for the bilateral filter
radius = 5
stdSpatial = 4
stdIntensity = 20

# Apply the bilateral filter to the grayscale image
filtered_gray_image = clean_Gaussian_noise_bilateral(gray_image, radius, stdSpatial, stdIntensity)

# Convert original color image to YCrCb color space
ycrcb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)

# Replace the Y (luma) channel with the filtered grayscale image
ycrcb_image[:, :, 0] = filtered_gray_image

# Convert back to BGR color space
restored_color_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)

# Visualize the original, filtered grayscale, and restored color images
plt.figure(figsize=(15, 10))
plt.subplot(131)
plt.title("Original Color Image")
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(132)
plt.title("Filtered Grayscale Image")
plt.imshow(filtered_gray_image, cmap='gray')
plt.axis("off")

plt.subplot(133)
plt.title("Restored Color Image")
plt.imshow(cv2.cvtColor(restored_color_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
