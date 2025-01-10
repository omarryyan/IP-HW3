# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''''
balls sol : 
radius = 5 
stdSpatial =  100
stdIntensity = 20

NoisyGrayImage sol : 
radius = 10
stdSpatial = 20
stdIntensity = 100

taj sol : 
radius = 5
stdSpatial = 100
stdIntensity = 20
 ## not sure that this is optimal tho. TODO://these params are optimal i think, we will go with those
  # same paramters looks good for  balls.jpg what do u think 
'''''

# Gaussian helper function
def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)

# cleans the image using bilateral filtering
def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):

    im = im.astype(np.float64)  # Ensure working with float64 for accuracy
    rows, cols = im.shape

    # Define the window size
    window_size = 2 * radius + 1

    # Pad the image to handle edges (padding with reflect mode)
    padded_im = np.pad(im, radius, mode='reflect')

    # Create spatial Gaussian kernel
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = gaussian(np.sqrt(x**2 + y**2), stdSpatial)

    # Normalize the spatial Gaussian kernel
    gs = gs / gs.sum()

    # Create a shifted view of the image for easy access to neighborhoods
    windows = np.lib.stride_tricks.sliding_window_view(padded_im, (window_size, window_size))
    windows = windows.reshape((rows, cols, window_size, window_size))

    # Calculate intensity differences
    intensity_diff = windows - im[:, :, np.newaxis, np.newaxis]

    # Compute intensity Gaussian
    gi = gaussian(intensity_diff, stdIntensity)

    # Combine spatial and intensity weights
    weights = gs * gi

    # Normalize the weights
    weights = weights / weights.sum(axis=(2, 3), keepdims=True)

    # Apply weights to the image and sum up
    filtered_im = np.sum(weights * windows, axis=(2, 3))

    return np.clip(filtered_im, 0, 255).astype(np.uint8) # Converting back to unit8


# Process multiple images
image_paths = ['balls.jpg', 'NoisyGrayImage.png', 'taj.jpg']

for image_path in image_paths:
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image directly

    if gray_image is None:
        print(f"Image file not found: {image_path}. Skipping.")
        continue

    # Define parameters based on the image
    if image_path == 'NoisyGrayImage.png':
        radius = 10
        stdSpatial = 20
        stdIntensity = 100
    else:  # Parameters for 'balls.jpg' and 'taj.jpg', works for both
        radius = 5
        stdSpatial = 150
        stdIntensity = 50

    # Apply the bilateral filter
    filtered_gray_image = clean_Gaussian_noise_bilateral(gray_image, radius, stdSpatial, stdIntensity)

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title(f"Original Grayscale Image ({image_path})")
    plt.imshow(gray_image, cmap='gray')
    plt.axis("off")

    plt.subplot(122)
    plt.title("Filtered Grayscale Image")
    plt.imshow(filtered_gray_image, cmap='gray')
    plt.axis("off")

    plt.show()
