# 导入必要的文件包
from imutils.video import VideoStream  # 读视频
import numpy as np
import argparse
import imutils  # 读视频
import time
import cv2
from rmn import RMN

# 初始化视频流
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

m = RMN()

# 循环读取视频流
while True:
    # 从线程视频流中获取帧并调整其大小，使其最大宽度为400像素
    frame = vs.read()
    # frame = imutils.resize(frame, width=400)
    #
    # # 转换成300*300*3的图片 输入模型中
    # (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    #                              (300, 300), (104.0, 177.0, 123.0))

    assert frame is not None
    results = m.detect_emotion_for_single_frame(frame)

    image = m.draw(frame, results)

    # 展示最终结果
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # 按下q结束程序
    if key == ord("q"):
        break

# 清除所有窗口
cv2.destroyAllWindows()
vs.stop()