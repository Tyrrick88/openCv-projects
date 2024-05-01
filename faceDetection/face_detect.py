import cv2
import numpy as np
import argparse

# here we construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# here the "--imqge" is the path of the Image
ap.add_argument("-i", "--image", required=True, help="Path to the Image ")
# here we pre-load the image to the pre_trained caffe model
ap.add_argument("-p", "__prototxt", required=True, help="Path to the pre-trained caffe Model")
# here i am adding the path to the caffe model
ap.add_argument("-m", "--model", required=True, help="path to the Caffe Pre-trained Model")

ap.add_argument("-c", "--confidence", required=True, help="minimizes false detections")
# Here we pass the Arguments.
args = vars(ap.parse_args(1))

# Now lets load the Model and createthe image blob.
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))