import numpy as np
import matplotlib.pyplot as plt
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/300.png"
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to isolate the plume (adjust threshold based on image intensity)
_, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Find contours of the plume
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour assuming it's the plume
plume_contour = max(contours, key=cv2.contourArea)

# Extract bounding box for the plume
x, y, w, h = cv2.boundingRect(plume_contour)

# Mask the plume for width measurement
plume_mask = np.zeros_like(gray_image)
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

