# import the necessary packages
# import centroidtracker
from shapedetector import ShapeDetector
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries
# ball in the HSV color space, then initialize the
# list of tracked points
colorLower = (0, 0, 245)
colorUpper = (255, 255, 255)
pts = deque(maxlen=args["buffer"])

video_capture = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# initialize data collection
frame_number = 0
height_data = {}
width_data = {}

while True:
    
    frame_number += 1
    height_data[frame_number] = []
    width_data[frame_number] = []
    
    # grab the current frame
    frame = video_capture.read()
    
    # use only the data from the tuple, not the meta info
    not_resized = frame[1]
    
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if not_resized is None:
        break
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(not_resized, width=350, height=350)
    ratio = not_resized.shape[0] / float(frame.shape[0])
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    # Shape detection of the white parts of the black mask
    # using the coordinates of the changes from white to black
    sd = ShapeDetector()
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        shape = sd.detect(c)
        
        # all data
        print("F: {}, H: {}, W: {}".format(frame_number, center[0], center[1]))
        
        # height data
        height_data[frame_number].append(center[0])
    
        # width data
        width_data[frame_number].append(center[0])
        
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    #time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# release the video capture
video_capture.release()

# close all windows
cv2.destroyAllWindows()

'''
# plot the data and show the graph for height - ax^2+bx+c = 0
fig = plt.figure()
axes=fig.add_subplot(111)
axes.plot(frame_data, height_data)
plt.title("All Balls - Height")
plt.show()

# Get data for one ball
new_x = []
new_y = []
for num, data in enumerate(graph_data_height):
    try:
        if data[0] != graph_data_height[num+1][0]:
            new_x.append(data[0])
            new_y.append(data[1])
    except IndexError:
        break

fig = plt.figure()
axes=fig.add_subplot(111)
axes.plot(new_x, new_y)
plt.title("One Ball - Height")
plt.show()

# plot the data and show the graph for width - ax^2+bx+c = 0
fig = plt.figure()
axes=fig.add_subplot(111)
axes.plot(frame_data, width_data)
plt.title("All Balls - Width")
plt.show()

# Get data for one ball
new_x = []
new_y = []
for num, data in enumerate(graph_data_width):
    try:
        if data[0] != graph_data_width[num+1][0]:
            new_x.append(data[0])
            new_y.append(data[1])
    except IndexError:
        break

fig = plt.figure()
axes=fig.add_subplot(111)
axes.plot(new_x, new_y)
plt.title("One Ball - Width")
plt.show()

# print tracking time
print("Tracked for {} seconds!".format(len(frame_data)/30))
'''