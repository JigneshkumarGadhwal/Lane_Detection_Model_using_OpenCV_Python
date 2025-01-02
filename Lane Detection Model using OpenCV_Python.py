### LANE DETECTION ON AN IMAGE FRAME:

# Importing necessary Modules:
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Defining canny function for image pre-processing:
def canny(image):
	# Converting an image in to a gray scale:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Applying Gaussian Blur on the image:
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	# Applying Canny Edge Detection Method:
	canny = cv2.Canny(blur, 50, 150)
	return canny


# Defining ROI function for selecting the lane region:
def region_of_interest(image):
	height = image.shape[0]
	# Determining the vertices of polygon:
	polygon = np.array([[(200, height), (1100, height), (550, 250)]])
	# Creating a black mask to blend it with the polygon region:
	mask = np.zeros_like(image)
	# Filling the mask with the polygon region:
	cv2.fillPoly(mask, polygon, 255)
	# Applying bitwise_and Operator to blend the image:
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


# Defining the display function to draw the line identified by the Hough Transform:
def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)

			# Drawing/displaying each line that we are itereating thorugh on the black mask image(line_image):
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return line_image


# Defining make_coordinates function to get the coordinates from the average slope and interception we determined:
def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * 3 / 5)
	# y = mx + b
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)

	return np.array([x1, y1, x2, y2])


# Defining average_slope_intercept function to make the detected and ploted lines smooth:
def average_slope_intercept(image, lines):
	left_fit = []  # Average line for left lane
	right_fit = []  # Average line for right lane
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Slope and intercept
			slope, intercept = parameters
			if slope < 0:
				left_fit.append((slope, intercept))
			else:
				right_fit.append((slope, intercept))

	# Handle empty cases
	left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
	right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None

	return np.array([line for line in [left_line, right_line] if line is not None])


# Loading an image:
image = cv2.imread(r"C:\Users\JERRY\Jupyter_Python_2K24_25\Computer Vision Projects\Lane detection\test_image.jpg")
# # Making a copy of an image:
# lane_image = np.copy(image)
# # Calling the canny function:
# canny_image = canny(lane_image)
# # Calling the ROI function:
# cropped_image = region_of_interest(canny_image)
# # Identifying the lines using Hough transform:
# lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# # Calling the average_slope_intercept funtion:
# average_line = average_slope_intercept(lane_image,lines)
# # Calling the display_lines function:
# line_image = display_lines(lane_image,average_line)
# # Now we combine/blend both the image (lane_image+line_image)
# combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

# # plt.figure(figsize=(10,10))
# # plt.imshow(gray,'gray')
# # plt.show()
# # print(f"Shape of an image: {gray.shape}")
# cv2.imshow("RoadImage",combo_image)
# cv2.waitKey(0)


## LANE DETECTION ON VIDEO FRAMES:

cap = cv2.VideoCapture(r"C:\Users\JERRY\Jupyter_Python_2K24_25\Computer Vision Projects\Lane detection\test2.mp4")

while cap.isOpened():
	ret, frame = cap.read()
	# Calling the canny function:
	canny_image = canny(frame)
	# Calling the ROI function:
	cropped_image = region_of_interest(canny_image)
	# Identifying the lines using Hough transform:
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	# Calling the display_lines function:
	line_image = display_lines(frame, lines)
	# Now we combine/blend both the image (lane_image+line_image)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
	cv2.imshow("RoadImage", combo_image)
	if cv2.waitKey(1) & 0xFF == (27):
		break
cap.release()
cv2.destroyAllWindows()
