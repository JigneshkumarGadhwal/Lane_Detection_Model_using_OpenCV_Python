# Lane Detection with OpenCV and Python

## Overview

This project demonstrates lane detection on both image and video frames using OpenCV and Python. The lane detection algorithm uses several techniques including image pre-processing (Canny edge detection), region of interest (ROI) masking, Hough transform for line detection, and averaging lines to smooth the output.

### Key Features:
- **Lane Detection on Image Frames**: Detect lanes in static images.
- **Lane Detection on Video Frames**: Detect lanes in real-time video streams.

---

## Table of Contents:
1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Explanation](#code-explanation)
4. [License](#license)

---

## Installation

To run the project, you will need the following Python libraries:
- **OpenCV**: For computer vision tasks.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing the output images and videos.

Install the dependencies using the following command:

```bash
pip install opencv-python numpy matplotlib
```

---

## Usage

### Lane Detection on Image:
The `canny` function is used to pre-process the image, applying edge detection using the Canny method. The region of interest (ROI) is defined using the `region_of_interest` function to limit the detection to the lane area. The Hough transform is used to detect the lane lines, and the results are blended with the original image.

### Lane Detection on Video:
The code captures frames from a video file, applies the same image pre-processing steps, and uses the same functions for detecting lane lines on each frame. The final result is displayed in real-time by overlaying the detected lanes on the video.

To detect lanes on an image:
```python
image = cv2.imread("path_to_image.jpg")
# Apply Canny edge detection
canny_image = canny(image)
# Apply ROI masking
cropped_image = region_of_interest(canny_image)
# Detect lane lines using Hough transform
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# Average and display lines
average_line = average_slope_intercept(image, lines)
line_image = display_lines(image, average_line)
# Combine the lane lines with the original image
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
cv2.imshow("Lane Detection", combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

To detect lanes on a video:
```python
cap = cv2.VideoCapture("path_to_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Apply Canny edge detection
    canny_image = canny(frame)
    # Apply ROI masking
    cropped_image = region_of_interest(canny_image)
    # Detect lane lines using Hough transform
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Average and display lines
    line_image = display_lines(frame, lines)
    # Combine the lane lines with the original frame
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Lane Detection", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Code Explanation

### 1. `canny(image)`:
This function converts the input image to grayscale, applies a Gaussian blur, and then uses the Canny edge detection algorithm to detect edges.

### 2. `region_of_interest(image)`:
This function defines a polygon-shaped region (ROI) to focus lane detection on the most relevant portion of the image.

### 3. `display_lines(image, lines)`:
This function iterates through the detected lines and draws them on a black image (line_image) to visualize the lane lines.

### 4. `make_coordinates(image, line_parameters)`:
Given the line parameters (slope and intercept), this function computes the start and end points of the line to plot.

### 5. `average_slope_intercept(image, lines)`:
This function averages the slopes and intercepts of detected lines to ensure the detected lanes are smoother and more reliable.

### 6. **Lane Detection on Video**:
The video lane detection process is similar to the image lane detection, but it is applied to each frame of the video. The processed frames are then displayed in real-time.

---

