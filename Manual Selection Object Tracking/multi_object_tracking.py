# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenEV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# initalize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# create lists for data used to plot a graph
x_axis = []
y_axis = [] 

vs = cv2.VideoCapture(args["video"])

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break
    
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=600)

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)

    # loop over the bounding boxes and draw them on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # gather data for graph
        x_axis.append(x)
        y_axis.append(y)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
        
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

        # if the 'q' key was pressed, break from the loop
    elif key == ord("q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()

# plot the data and show the graph
fig= plt.figure()
axes=fig.add_subplot(111)
axes.plot(x_axis, y_axis)
plt.show()