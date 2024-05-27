import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Red points are dialectal points where "个" has demonstrative semantics.
# Black points are dialectal points where "个" undergoes semantic bleaching.

def extract_color_range(image_path):# Extract the color range of two different kinds of points
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    lower_bound = np.array([np.min(h), np.min(s), np.min(v)])
    upper_bound = np.array([np.max(h), np.max(s), np.max(v)])
    return lower_bound, upper_bound

# Measure time for each step
start_time = time.time()

# Load the color reference images
red_image_path = r"D:\Red points.png"# The color of red points
black_image_path = r"D:\Black points.png"# The color of black points

# Extract color ranges for red and black points
red_lower, red_upper = extract_color_range(red_image_path)
black_lower, black_upper = extract_color_range(black_image_path)
print(f"Extracted color ranges in {time.time() - start_time:.2f} seconds")

# Load the main map image
image_path = r"D:\picture.png"
print(f"Loading image from: {image_path}")

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print(f"Error: Image could not be loaded from {image_path}")
    exit()
print(f"Loaded main image in {time.time() - start_time:.2f} seconds")

# Convert the image to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print(f"Converted image to HSV in {time.time() - start_time:.2f} seconds")

# Mask the red points
red_mask = cv2.inRange(image_hsv, red_lower, red_upper)
red_points = cv2.findNonZero(red_mask)

# Mask the black points
black_mask = cv2.inRange(image_hsv, black_lower, black_upper)
black_points = cv2.findNonZero(black_mask)
print(f"Masked points in {time.time() - start_time:.2f} seconds")

# Convert points to a suitable format
red_points = np.squeeze(red_points) if red_points is not None else np.array([])
black_points = np.squeeze(black_points) if black_points is not None else np.array([])

# Print the number of points detected
print(f"Number of red points detected: {len(red_points)}")
print(f"Number of black points detected: {len(black_points)}")

# Visualize the masks and points
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Red mask
plt.subplot(1, 3, 2)
plt.imshow(red_mask, cmap='gray')
plt.title('Red Mask')

# Black mask
plt.subplot(1, 3, 3)
plt.imshow(black_mask, cmap='gray')
plt.title('Black Mask')

plt.show()
print(f"Visualized masks in {time.time() - start_time:.2f} seconds")

# Check if there are enough points to proceed
if red_points.size == 0 or black_points.size == 0:
    print("Error: Not enough points detected to fit the model.")
    exit()

# Combine points and create labels
X = np.vstack((red_points, black_points))
y = np.hstack((np.ones(len(red_points)), np.zeros(len(black_points))))
print(f"Prepared data for model in {time.time() - start_time:.2f} seconds")

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)
print(f"Trained Logistic Regression model in {time.time() - start_time:.2f} seconds")

# Get the separating line
coef = model.coef_[0]
intercept = model.intercept_[0]
x_plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_plot = -(coef[0] * x_plot + intercept) / coef[1]

# Plot the results
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Red Points')
plt.scatter(black_points[:, 0], black_points[:, 1], color='black', label='Black Points')
plt.plot(x_plot, y_plot, 'k-', label='Division Line')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()
print(f"Plotted results in {time.time() - start_time:.2f} seconds")

# Total time
print(f"Total processing time: {time.time() - start_time:.2f} seconds")
