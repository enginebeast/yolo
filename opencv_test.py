import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# OpenCV 프레임워크 라이브러리
import cv2

# Reading boundingbox
box = pd.read_csv("./cardataset/train_solution_bounding_boxes (1).csv")
box.info()
box

# Writing boundingbox
sample1 = cv2.imread("./cardataset/training_images/vid_4_1000.jpg")
sample1

# Define color channel sequence
sample = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)
sample

# Extract xy coordination
point = box.iloc[0]
point, type(point)

# Extract start coordinate and end coordinate
pt1 = (int(point.xmin), int(point.ymin))
pt2 = (int(point.xmax), int(point.ymax))
### Another way
# pt1 = (point["xmin"], point["ymin"])
# pt2 = (point["xmax"], point["ymax"])
pt1, pt2

# Drawing boundingbox(Just given coordinate)
cv2.rectangle(sample, pt1, pt2, color=(255, 0, 0), thickness=2)

# Show image with bounding box
plt.imshow(sample)
plt.show()

# 이미지 불러오기
sample2 = cv2.imread("./cardataset/training_images/vid_4_10000.jpg")

# RGB 순서로 변경하기
sample2_rgb = cv2.cvtColor(sample2, cv2.COLOR_BGR2RGB)

# 해당 이미지 파일에 대한 좌표값 추출하기
# point2 = box[box["image"]=="vid_4_10000.jpg"]
point2 = box.iloc[1]

# 시작좌표와 종료좌표 생성하기
pt1_2 = (int(point2.xmin), int(point2.ymin))
pt2_2 = (int(point2.xmax), int(point2.ymax))

# 원본 이미지에 바운딩박스 그리기"""
cv2.rectangle(sample2_rgb, pt1_2, pt2_2, color=(255, 250, 0), thickness=2)

# 이미지에 바운딩 박스를 그리고 잘 보이는지 다시 imshow()하기
plt.imshow(sample2_rgb)
plt.show()