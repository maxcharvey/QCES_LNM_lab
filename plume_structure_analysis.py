import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path1 = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/300.png"
image_path2 = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/550.png"
image_path3 = "/Users/maxharvey/Documents/University/Part_III/I_Core_Courses/III_Laboratory_and_numerical_methods/Processed_files/800.png"


# List of image paths (you can add more image paths as needed)
image_paths = [image_path1, image_path2, image_path3]

# Could edit this if the threshold values need to be different for each image
threshold_values = []

# To store the plume widths in
plume_widths_list = []

for image_path in image_paths:

    image = cv2.imread(image_path)

    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the Region of Interest (ROI) (x, y, width, height)
    roi_x, roi_y, roi_w, roi_h = 200, 43, 150, 140  # Adjust these values as needed

    # Extract the ROI from the grayscale image
    roi = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Plot the histogram of pixel intensities in the ROI
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid for subplots

    # Plot 1: ROI Visualization
    axes[0, 0].imshow(cv2.cvtColor(cv2.rectangle(gray_image.copy(), (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Region of Interest (ROI)")
    axes[0, 0].axis("off")

    # Plot 2: Histogram of Pixel Intensities in ROI
    axes[0, 1].hist(roi.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[0, 1].set_title('Histogram of Pixel Intensities in ROI')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid()

    # Manual Threshold Selection
    manual_threshold = 155  # Replace this with your chosen threshold value

    # Apply the manually selected threshold
    _, thresholded = cv2.threshold(roi, manual_threshold, 255, cv2.THRESH_BINARY)

    # Overlay the thresholded region on the ROI
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert ROI to color for overlay
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours of the detected plume
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=1)

    # Plot 3: Thresholded Region Overlayed on the ROI
    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Thresholded Region Overlayed (Threshold = {manual_threshold})")
    axes[1, 0].axis("off")

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

    # Plot 4: Plume Width as a Function of Depth
    depth = np.arange(len(plume_widths))
    axes[1, 1].plot(depth, plume_widths)
    axes[1, 1].set_xlabel('Depth (pixels)')
    axes[1, 1].set_ylabel('Plume Width (pixels)')
    axes[1, 1].set_title('Plume Width as a Function of Depth')
    axes[1, 1].grid()

    plume_widths_list.append(plume_widths)

    # Adjust layout to prevent overlap and show all subplots
    plt.tight_layout()
    #plt.show()

# Now we need to manually work out which bits of the graph are the correct linear region so that we can try and work out the gradient
# of the straight line section of the graph which corresponds to the actual correct plume profile in the experiment 

selection_regions = [[15,55], [15,62], [20,50]] # These are the regions that I have selected for each of the images
m_values = []
c_values = []
print(selection_regions[0][0])

fig, ax2 = plt.subplots()
for i in range(3):
    x = np.arange(selection_regions[i][0], selection_regions[i][1])
    y = plume_widths_list[i][selection_regions[i][0]:selection_regions[i][1]]
    # we also want to calulate the gradient of the best fit line for the series we are plotting
    m, c = np.polyfit(x, y, 1)
    m_values.append(m)
    c_values.append(c)
    ax2.plot(x, m*x + c, label=f"Best fit line for image {i+1}")
    ax2.plot(x, y, label=f"Image {i+1}")


plt.show()
print(np.array(m_values)*(5/6))
print(c_values)

