#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import cv2

from ultralytics import YOLOv10



# Open the video file
# video_path = "images/resources/demo.mp4"
video_path = "D:/data/video/istockphoto-2154830401-640_adpp_is.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"视频文件 {video_path} 不存在。")

# Load the YOLOv8 model
# model = YOLOv10("yolov8n.pt")
model = YOLOv10("../models_pt/yolov10x.pt")



cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    raise IOError(f"无法打开视频文件 {video_path}")



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # 检查推理结果
        if results:
            # 在帧上可视化推理结果
            annotated_frame = results[0].plot()

            # 显示注释过的帧
            cv2.imshow("YOLOv10 Inference", annotated_frame)

            # # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            # # Display the annotated frame
            # cv2.imshow("YOLOv10 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()