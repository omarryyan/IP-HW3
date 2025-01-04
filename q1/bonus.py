import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "1.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Load the target image for comparison
target_image_path = "image_1.jpg"
target_image = cv2.imread(target_image_path)
if target_image is None:
    raise FileNotFoundError(f"Target image file '{target_image_path}' not found.")

# Convert images to RGB (OpenCV loads images in BGR format by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

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

# 1. Original Image
titles = ["Original"]
images = [image_rgb]
comparisons = []

# 2. Grayscale Conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
min_max_gray = apply_min_max_filter(gray)  # Apply Min-Max normalization to grayscale
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
canny_edges = cv2.Canny(gray, 100, 200)
min_max_canny = apply_min_max_filter(canny_edges)  # Apply Min-Max normalization to Canny edge detection
comparisons.append(("Min-Max Canny Edge Detection", compare_images(min_max_canny, target_image)))
titles.append("Min-Max Canny Edge Detection")
images.append(min_max_canny)

# 6. Sobel Filter (X and Y)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
min_max_sobel_combined = apply_min_max_filter(sobel_combined)  # Apply Min-Max normalization to Sobel combined
comparisons.append(("Min-Max Sobel Combined", compare_images(min_max_sobel_combined, target_image)))
titles.append("Min-Max Sobel Combined")
images.append(min_max_sobel_combined)

# 7. Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
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
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
min_max_adaptive_thresh = apply_min_max_filter(adaptive_thresh)  # Apply Min-Max normalization to Adaptive thresholding
comparisons.append(("Min-Max Adaptive Thresholding", compare_images(min_max_adaptive_thresh, target_image)))
titles.append("Min-Max Adaptive Thresholding")
images.append(min_max_adaptive_thresh)

# 10. Histogram Equalization
equalized = cv2.equalizeHist(gray)
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

# Display all the results
display_images(titles, images)

# Print comparison results
print("Filter Comparison Results:")
for filter_name, diff in comparisons:
    print(f"{filter_name}: Difference = {diff}")

# Suggest the best match
best_match = min(comparisons, key=lambda x: x[1])
print(f"\nBest matching filter: {best_match[0]} with a difference of {best_match[1]}")
