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

    # Normalize gi to avoid any scaling issue
    gi = gi / gi.sum(axis=(2, 3), keepdims=True)

    # Apply the spatial and intensity Gaussian together
    weights = gs * gi

    # Normalize the combined weight matrix
    weights = weights / weights.sum(axis=(2, 3), keepdims=True)

    # Multiply the window values with the weights and sum them
    filtered_im = np.sum(weights * windows, axis=(2, 3))

    # Convert the result back to uint8
    return np.clip(filtered_im, 0, 255).astype(np.uint8)


# Example Usage
original_image_path = 'NoisyGrayImage.png'  # Replace with your actual file name
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded successfully
if image is None:
    raise FileNotFoundError("Image file not found. Ensure it is in the same folder as this script.")

# Parameters for the bilateral filter
radius = 10
stdSpatial = 10.0
stdIntensity =150

# Apply the bilateral filter
filtered_image = clean_Gaussian_noise_bilateral(image, radius, stdSpatial, stdIntensity)

# Visualize the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(122)
plt.title("Filtered Image")
plt.imshow(filtered_image, cmap='gray')
plt.axis("off")

plt.show()
