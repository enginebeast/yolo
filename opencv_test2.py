import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# OpenCV Framework library
import cv2


net = cv2.dnn.readNet("./config/yolov3.weights", "./config/yolov3.cfg")

# Read the labeling name
classes = []
with open("./config/coco.names", "r") as f :
    classes = [line.strip() for line in f.readlines()]

# Read the layer
layer_names = net.getLayerNames()

# Extract unconnected outlayer
output_layer = [layer_names[ i-1] for i in net.getUnconnectedOutLayers()]

# Check opence_test.py
img = cv2.imread("./cardataset/training_images/vid_4_10000.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, channels = img.shape

# Change the image as a blob and normalize
# 1/ 256 : Devide image by 256 pixels
# (416, 416) : Normalize horizontal and vertical size
# (0, 0, 0) : Change the image channel as a black
# swapRB: Change RGB from BGR
# crop: crop the image
blob = cv2.dnn.blobFromImage(img,
                            1/256,
                            (416, 416),
                            (0, 0, 0),
                            swapRB = True,
                            crop = False)

# Feed the data to YOLO
net.setInput(blob)
# Get the result
outs = net.forward(output_layer)
