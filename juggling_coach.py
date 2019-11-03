from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt

number_of_balls = 5

ap = argparse.ArgumentParser()

ap.add_argument(
    "-v",
    "--video",
    help="path to the video file"
)

args = vars(ap.parse_args())

# Adjust these values based on the color of the object
# you want to track. You will create a black and white "mask",
# where the white parts are going to be what you want to track.
# later you will use the findContours method of cv2 to detect
# the black and white edges in the mask. Once you have this,
# you can find the perimeter of every white shape in the mask, 
# and find the center most point in the shape, the centroid.
colorLower = (0, 0, 245)
colorUpper = (255, 255, 255)

# feed the video into cv2
video_capture = cv2.VideoCapture(args["video"])

# wait for video file to warm up / load in
time.sleep(2.0)

# initialize data collection
frame_number = 0
pattern_data = {}

while True:
    
    frame_number += 1
    frame_data = video_capture.read()
    not_resized = frame_data[1]
    
    # if there is not data for a frame, then the video has ended
    if not_resized is None:
        break
    
    # initialize data collection for the frame
    pattern_data[frame_number] = []
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(not_resized, width=350, height=350)
    ratio = not_resized.shape[0] / float(frame.shape[0])
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct the black and white mask, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # uncomment the next line to see the black and white mask
    #cv2.imshow("mask", mask)
    
    # find the white shapes in the mask
    edges = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    white_shapes = imutils.grab_contours(edges)
    
    # based on the data of the coordinates of the edges (aka
    # changes from white to black) in the mask, find the center
    # most part of each the white shape found in the mask
    centroid = None
    for white_shape in white_shapes:
        M = cv2.moments(white_shape)
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # collect the pattern data found in each frame (height, width)
        pattern_data[frame_number].append((centroid[0], centroid[1]))
        
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the edges of the white shapes found in the mask
        # onto the regular non-mask video frame. Also draw a red dot
        # to represent each centroid found.
        white_shape = white_shape.astype("float")
        white_shape = white_shape.astype("int")
        cv2.drawContours(frame, [white_shape], -1, (0, 255, 0), 2)
        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
    
    else:
        # adjust the number below for the number of balls you're tracking
        # sometimes not all of the balls in a frame are detected due to
        # the color threshold values and the black white portions of the mask
        if len(pattern_data[frame_number]) == number_of_balls:
            print("Frame {}: {}".format(frame_number, pattern_data[frame_number]))
        else:
            # toss out any frame where not all the balls were detected
            del pattern_data[frame_number]
    
    # comment the next line if you want the data without showing the video
    cv2.imshow("Frame", frame)
    
    # uncomment the next line if you want the video in slow motion
    #time.sleep(0.1)
    
    # if the 'q' key is pressed, stop the video
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# release the video capture
video_capture.release()

# close all windows
cv2.destroyAllWindows()

'''
# plot the data and show the graphs
fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(list(pattern_data.keys()), pattern_data.values()[0])
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