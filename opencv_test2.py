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


# List storing object class ids
class_ids = []
# List storing object confidence scores
confidences = []
# List storing object bounding box coordinates
boxes = []

# structure of detection
# 0th index : x center coordinate
# 1st index : y center coordinate
# 2nd index : width of bounding box
# 3rd index : height of bounding box
# 4th index : confidence score
# After 5th : probability value
for out in outs :
    for detection in out :
        scores = detection[5:]
        # Find an index which has max score
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 3rd tab
        if confidence > 0.5 :
            # Return absolute coordinate
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            # Return relative width and height
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            # Caculate starting coordinate
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            # Remove overlapping bounding box
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

 
# If there is recognized object
if len(indexes) > 0:
    print(indexes.flatten())
    for i in indexes.flatten() : 
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = str(round(confidences[i], 2))
        #Color of bounding box
        color = colors[i]

        # Draw bounding box
        # Final number 2 is a number for line thickness of bounding box
        cv2.rectangle(img, (x, y), ((x+w),(y+h)), color, 2)
        # Print label and confidence
        cv2.putText(img, label + " " + confidence, (x, y-20), font, 2, (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()

else :
    print("There is not recognized object!!!")