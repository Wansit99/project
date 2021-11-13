import cv2
import numpy as np

# 读入一张RGB图片
img = cv2.imread(r'Cell.BMP', cv2.IMREAD_COLOR)

# 将图片转为灰度图片来分析
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图片来分析

# 处理了两次,为消除细胞体内的浓淡深浅差别,避免在后续运算中产生孔洞
# 高斯滤波(低通,去掉高频分量,图像平滑)
gray1 = cv2.GaussianBlur(gray, (3, 3), 0)
gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)



# 0~255反相,为了适应腐蚀算法
gray2 = 255 - gray1

# 自动阈值处理
ret, thresh = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# 下面为了去除细胞之间的粘连,以免把两个细胞计算成一个
# 进行腐蚀操作,kernel=初值为1的2*2数组
kernel = np.ones((2, 2), np.uint8)
# 腐蚀:卷积核沿着图像滑动，如果与卷积核对应的原图像的所有像素值都是1，那么中心元素就保持原来的像素值，否则就变为零。
erosion = cv2.erode(thresh, kernel, iterations=3)


#获得轮廊
contours, hirearchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#建立空list，存放筛选后的结果
contours_out=[]
for i in contours:
     if (cv2.contourArea(i)>100):
         # 排除面积不足“100"的细胞
        contours_out.append(i)


# 展示结果
img2 = img.copy()
# 用红色线,描绘块轮廓
cv2.drawContours(img2, contours_out, -1, (50, 50, 250), 2)

#保存结果
for i in range(0,len(contours_out)):
    x, y, w, h = cv2.boundingRect(contours_out[i])
    cropImg = img[y: y + h, x:x + w]
    cv2.imwrite("./result/" + str(i) + ".jpg", cropImg)


cv2.imshow("result", img2)
cv2.waitKey()

