# Omar_Ryyan, 322283755
# Abdulrahman_Assi, 212371421

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import grayscale

# Load the image
image_path = "1.jpg"
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Load the target image for comparison
target_image_path = "image_4.jpg"
target_image = cv2.imread(target_image_path,cv2.IMREAD_GRAYSCALE)
if target_image is None:
    raise FileNotFoundError(f"Target image file '{target_image_path}' not found.")

# Convert images to RGB (OpenCV loads images in BGR format by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

def y_axis_average_filter(image, radius):

    # Pad the image along the Y-axis to handle edge cases
    padded_image = np.pad(image, ((radius, radius), (0, 0)), mode='constant', constant_values=0)

    # Create an empty image for the result
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Apply the Y-axis average filter
    for y in range(image.shape[0]):
        # Sum over the window centered at the current pixel along the Y-axis
        filtered_image[y, :] = np.mean(padded_image[y:y + 2 * radius + 1, :], axis=0)

    # Clip the result to the valid range [0, 255] and convert to uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image


def vertical_edge_detection_laplacian(image, ksize=3):
    """
    Apply a Laplacian filter for vertical edge detection.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        ksize (int): Kernel size for the Laplacian filter. Must be an odd number (1, 3, 5, ...).

    Returns:
        numpy.ndarray: Image with vertical edges detected.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Apply the Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)

    # Convert the result to an absolute scale
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    return abs_laplacian

# Function to display images
def display_images(titles, images):
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols  # Calculate rows dynamically
    plt.figure(figsize=(15, 5 * rows))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to apply shearing
def apply_shearing(image, shear_x=0.2, shear_y=0.0):
    rows, cols, ch = image.shape
    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    return sheared_image

# Function to compare images
def compare_images(img1, img2):
    # Ensure both images are resized to the same dimensions
    img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Convert images to the same number of channels
    if len(img1_resized.shape) == 2:  # If img1 is grayscale
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # If img2 is grayscale
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Compute the absolute difference
    difference = cv2.absdiff(img1_resized, img2)
    # Return the sum of the difference as a measure of similarity
    return np.sum(difference)

# Function to apply Min-Max normalization
def apply_min_max_filter(image):
    # Normalize the image to the range [0, 255]
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    return np.uint8(normalized_image)


def swap_image_halves(image):
    """
    Swap the upper and lower halves of an image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with upper and lower halves swapped.
    """
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Ensure the height is divisible by 2
    if height % 2 != 0:
        raise ValueError("Image height must be divisible by 2 for swapping halves.")

    # Split the image into two halves
    upper_half = image[:height // 2, :]
    lower_half = image[height // 2:, :]

    # Swap the halves
    swapped_image = np.vstack((lower_half, upper_half))
    return swapped_image

# 1. Original Image
titles = ["Original"]
images = [image_rgb]
comparisons = []

# 2. Grayscale Conversion
min_max_gray = apply_min_max_filter(image)  # Apply Min-Max normalization to grayscale
comparisons.append(("Min-Max Grayscale", compare_images(min_max_gray, target_image)))
titles.append("Min-Max Grayscale")
images.append(min_max_gray)

# 3. Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image_rgb, (15, 15), 0)

min_max_gaussian_blur = apply_min_max_filter(gaussian_blur)  # Apply Min-Max normalization to Gaussian blur
comparisons.append(("Min-Max Gaussian Blur", compare_images(min_max_gaussian_blur, target_image_rgb)))
titles.append("Min-Max Gaussian Blur")
images.append(min_max_gaussian_blur)

# 4. Median Blur
median_blur = cv2.medianBlur(image_rgb, 15)
min_max_median_blur = apply_min_max_filter(median_blur)  # Apply Min-Max normalization to Median blur
comparisons.append(("Min-Max Median Blur", compare_images(min_max_median_blur, target_image_rgb)))
titles.append("Min-Max Median Blur")
images.append(min_max_median_blur)

# 5. Edge Detection (Canny)
canny_edges = cv2.Canny(image, 100, 200)
min_max_canny = apply_min_max_filter(canny_edges)  # Apply Min-Max normalization to Canny edge detection
comparisons.append(("Min-Max Canny Edge Detection", compare_images(min_max_canny, target_image)))
titles.append("Min-Max Canny Edge Detection")
images.append(min_max_canny)

# 6. Sobel Filter (X and Y)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
min_max_sobel_combined = apply_min_max_filter(sobel_combined)  # Apply Min-Max normalization to Sobel combined
comparisons.append(("Min-Max Sobel Combined", compare_images(min_max_sobel_combined, target_image)))
titles.append("Min-Max Sobel Combined")
images.append(min_max_sobel_combined)

# 7. Laplacian Filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
min_max_laplacian = apply_min_max_filter(laplacian)  # Apply Min-Max normalization to Laplacian
comparisons.append(("Min-Max Laplacian", compare_images(min_max_laplacian, target_image)))
titles.append("Min-Max Laplacian")
images.append(min_max_laplacian)

# 8. Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image_rgb, 15, 75, 75)
min_max_bilateral = apply_min_max_filter(bilateral_filter)  # Apply Min-Max normalization to Bilateral filter
comparisons.append(("Min-Max Bilateral Filter", compare_images(min_max_bilateral, target_image_rgb)))
titles.append("Min-Max Bilateral Filter")
images.append(min_max_bilateral)

# 9. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
min_max_adaptive_thresh = apply_min_max_filter(adaptive_thresh)  # Apply Min-Max normalization to Adaptive thresholding
comparisons.append(("Min-Max Adaptive Thresholding", compare_images(min_max_adaptive_thresh, target_image)))
titles.append("Min-Max Adaptive Thresholding")
images.append(min_max_adaptive_thresh)

# 10. Histogram Equalization
equalized = cv2.equalizeHist(image)
min_max_equalized = apply_min_max_filter(equalized)  # Apply Min-Max normalization to Histogram equalization
comparisons.append(("Min-Max Histogram Equalization", compare_images(min_max_equalized, target_image)))
titles.append("Min-Max Histogram Equalization")
images.append(min_max_equalized)

# 11. Shearing
sheared_image = apply_shearing(image_rgb, shear_x=0.2, shear_y=0.1)
min_max_shearing = apply_min_max_filter(sheared_image)  # Apply Min-Max normalization to Shearing
comparisons.append(("Min-Max Shearing", compare_images(min_max_shearing, target_image_rgb)))
titles.append("Min-Max Shearing")
images.append(min_max_shearing)

# 12. Convolution
convolution_result = cv2.filter2D(image_rgb, -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
min_max_convolution = apply_min_max_filter(convolution_result)  # Apply Min-Max normalization to Convolution
comparisons.append(("Min-Max Convolution", compare_images(min_max_convolution, target_image_rgb)))
titles.append("Min-Max Convolution")
images.append(min_max_convolution)

def apply_row_mean_normalization(image):
    rows, cols, ch = image.shape
    for row in range(rows):
        # Calculate the mean of the row
        row_mean = np.mean(image[row])
        # Replace every pixel in the row with the row's mean value
        image[row] = row_mean
    return image

# Apply row-wise mean normalization to the RGB image (a copy to keep original intact)
normalized_image_rgb = apply_row_mean_normalization(image_rgb.copy())
cv2.imwrite("normalized_image_rgb.jpg", normalized_image_rgb)

# Display original image and normalized image
titles.append("Row-wise Mean Normalized Image")
images.append(normalized_image_rgb)

# Display all the results
display_images(titles, images)

# Swap the upper and lower halves
swapped_image = swap_image_halves(image)

cv2.imshow("swap image",swapped_image)
cv2.waitKey(0)

# Print comparison results
print("Filter Comparison Results:")
for filter_name, diff in comparisons:
    print(f"{filter_name}: Difference = {diff}")

# Suggest the best match
best_match = min(comparisons, key=lambda x: x[1])
print(f"\nBest matching filter: {best_match[0]} with a difference of {best_match[1]}")


'''
need to do : 
image 4 
image 5
image 7 
image 8 
image 9 
'''