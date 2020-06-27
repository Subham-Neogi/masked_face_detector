from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from MaskDetector import MaskDetector
# constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-s", "--saved", required=True, help="path to saved model")
ap.add_argument("-k", "--keep", default="n", help="save frames in secondary storage: Options [y/n]")
ap.add_argument("-o", "--output", help="path to output directory", default="./results")
ap.add_argument("-c", "--confidence", type=float, default=0.20, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the video stream
vs = VideoStream(src=0).start() #  a video file (rather than a video stream) swap out the VideoStream  class for FileVideoStream
time.sleep(15.0)
print("[INFO] Starting video stream...")

label = ['NO MASK', 'MASK']
labelColor = [(10, 0, 255), (10, 255, 0)]

mask_detector = MaskDetector()
print("[INFO] Loading saved model...")
mask_detector.create_model(args["saved"])

count=0

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    count+=1
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi_color = frame[startY:endY, startX:endX]
            mask_detector.decode_img(roi_color)
            predicted = mask_detector.predict()
            predicted = np.argmax(predicted[0])
            print("[INFO] Predicted: {}".format(label[predicted]))
            #predicted = 0
            # draw the bounding box of the face along with the associated
            # probability
            text = "{} {:.2f}%".format(label[predicted],confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), labelColor[predicted], 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, labelColor[predicted], 2)
    # save images
    if (args["keep"].lower() == 'y'):
        outputPath = os.path.sep.join([args["output"], str(count) + '.jpg'])
        cv2.imwrite(outputPath, frame)
    # show the output frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# python face_detection_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel