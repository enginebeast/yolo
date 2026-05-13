import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# OpenCV Framework library
import cv2

# Reading boundingbox
box = pd.read_csv("./cardataset/train_solution_bounding_boxes (1).csv")
box.info()

# Writing boundingbox
sample1 = cv2.imread("./cardataset/training_images/vid_4_1000.jpg")

# Define color channel sequence
sample = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)

# Extract xy coordination
point = box.iloc[0]

# Extract start coordinate and end coordinate
pt1 = (int(point.xmin), int(point.ymin))
pt2 = (int(point.xmax), int(point.ymax))
### Another way
# pt1 = (point["xmin"], point["ymin"])
# pt2 = (point["xmax"], point["ymax"])

# Drawing boundingbox(Just given coordinate)
cv2.rectangle(sample, pt1, pt2, color=(255, 0, 0), thickness=2)

# Show image with bounding box
plt.imshow(sample)
plt.show()

# Retry it 
sample2 = cv2.imread("./cardataset/training_images/vid_4_10000.jpg")

sample2_rgb = cv2.cvtColor(sample2, cv2.COLOR_BGR2RGB)

point2 = box.iloc[1]

pt1_2 = (int(point2.xmin), int(point2.ymin))
pt2_2 = (int(point2.xmax), int(point2.ymax))
cv2.rectangle(sample2_rgb, pt1_2, pt2_2, color=(255, 250, 0), thickness=2)

plt.imshow(sample2_rgb)
plt.show()