# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

# 导入必要的文件包
from imutils.video import VideoStream  # 读视频
import numpy as np
import argparse
import imutils  # 读视频
import time
import cv2

# 从命令行读入必要的参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 读取模型权重
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 初始化视频流
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 循环读取视频流
while True:
    # 从线程视频流中获取帧并调整其大小，使其最大宽度为400像素
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # 转换成300*300*3的图片 输入模型中
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 将图像送入模型中
    net.setInput(blob)
    detections = net.forward()

    # 统计抬头的人数
    total_up = 0
    # 总人脸数
    total = 0

    # 读取检测出的每一个人脸
    for i in range(0, detections.shape[2]):
        # 提取出模型计算出的概率
        confidence = detections[0, 0, i, 2]

        # 如果概率小于0.5 则认为是误判，不属于人脸
        if confidence < args["confidence"]:
            continue
        # 已经检测到人脸
        total += 1
        # 计算出人脸的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # 框出人脸
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)


    # 展示最终结果
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 当未检测到人脸时，默认抬头率为0
    if total == 0:
        print("None one")
    else:
        print("Someone here")

    # 按下q结束程序
    if key == ord("q"):
        break


# 清除所有窗口
cv2.destroyAllWindows()
vs.stop()
