import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/300.png"
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the Region of Interest (ROI) (x, y, width, height)
roi_x, roi_y, roi_w, roi_h = 200, 43, 150, 140  # Adjust these values as needed

# Visualize the ROI on the original image
roi_visual = image.copy()
roi_visual = gray_image.copy()
cv2.rectangle(roi_visual, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

# Display the image with the ROI highlighted
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(roi_visual, cv2.COLOR_BGR2RGB))
plt.title("Region of Interest (ROI)")
plt.axis("off")
plt.show()

# Extract the ROI from the grayscale image
roi = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# Dynamic thresholding using Otsu's method
_, thresholded = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Alternatively, adaptive thresholding (comment out Otsu's thresholding above if using this)
#thresholded = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, blockSize=11, C=2)

# Display the thresholded image
plt.figure(figsize=(10, 6))
plt.imshow(thresholded, cmap='gray')
plt.title("Thresholded Image (Dynamic Thresholding)")
plt.axis("off")
plt.show()

# Find contours of the plume
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour assuming it's the plume
plume_contour = max(contours, key=cv2.contourArea)

# Extract bounding box for the plume within the ROI
x, y, w, h = cv2.boundingRect(plume_contour)

# Mask the plume for width measurement
plume_mask = np.zeros_like(roi)
cv2.drawContours(plume_mask, [plume_contour], -1, 255, thickness=cv2.FILLED)

# Calculate the width of the plume at each depth (row)
plume_widths = []
for row in range(y, y + h):
    row_data = plume_mask[row, x:x + w]
    plume_pixels = np.where(row_data == 255)[0]
    if len(plume_pixels) > 0:
        plume_width = plume_pixels[-1] - plume_pixels[0]
    else:
        plume_width = 0
    plume_widths.append(plume_width)

# Plot the plume width as a function of depth
depth = np.arange(len(plume_widths))
plt.figure(figsize=(10, 6))
plt.plot(plume_widths, depth)
plt.xlabel('Depth (pixels)')
plt.ylabel('Plume Width (pixels)')
plt.title('Plume Width as a Function of Depth')
plt.grid()
plt.show()

