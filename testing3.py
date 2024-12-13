import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/800.png"
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the Region of Interest (ROI) (x, y, width, height)
roi_x, roi_y, roi_w, roi_h = 200, 43, 150, 140  # Adjust these values as needed

# Visualize the ROI on the original image
roi_visual = gray_image.copy()
cv2.rectangle(roi_visual, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

# Display the image with the ROI highlighted
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(cv2.cvtColor(roi_visual, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
plt.title("Region of Interest (ROI)")
plt.axis("off")
plt.show()

# Extract the ROI from the grayscale image
roi = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# Plot the histogram of pixel intensities in the ROI
plt.figure(figsize=(10, 6))
plt.hist(roi.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title('Histogram of Pixel Intensities in ROI')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Manual Threshold Selection
manual_threshold = 155  # Replace this with your chosen threshold value

# Apply the manually selected threshold
_, thresholded = cv2.threshold(roi, manual_threshold, 255, cv2.THRESH_BINARY)

# Overlay the thresholded region on the ROI
overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert ROI to color for overlay
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours of the detected plume
cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=1)

# Display the thresholded region overlayed on the ROI
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Thresholded Region Overlayed (Threshold = {manual_threshold})")
plt.axis("off")
plt.show()

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
plt.plot(depth, plume_widths)
plt.xlabel('Depth (pixels)')
plt.ylabel('Plume Width (pixels)')
plt.title('Plume Width as a Function of Depth')
plt.grid()
plt.show()

