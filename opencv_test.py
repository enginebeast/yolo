import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# OpenCV 프레임워크 라이브러리
import cv2

# Reading boundingbox
box = pd.read_csv("./cardataset/train_solution_bounding_boxes (1).csv")
box.info()
print(box)

# Writing boundingbox
sample1 = cv2.imread("./cardataset/training_images/vid_4_1000.jpg")
print(sample1)

# Define color sequence
sample = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)
print(sample)

