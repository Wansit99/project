import cv2 as cv
import numpy as np

class LaneLineDetection:
    def __init__(self):
        # leftline 和rightline车道检测的两条线
        # 每一条线分别有两个点决定
        self.left_line = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        self.right_line = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}

    def process(self, frame, method=0):
        # 将图像转化为灰度图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # canny边缘检测
        binary = cv.Canny(gray, 150, 300)
        h, w = gray.shape
        # 根据车道线的限值 可以忽略一定区域
        binary[0:int(h/2+40),0:w] = 0
        # 轮廓查找
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 创建输出用的空白图像
        out_image = np.zeros((h, w), frame.dtype)
        # 遍历每一个轮廓，进行轮廓分析
        for cnt in range(len(contours)):
            # 通过多种特征筛选
            p = cv.arcLength(contours[cnt], True)
            # 计算轮廓面积
            area = cv.contourArea(contours[cnt])
            # 获取轮廓的中心坐标以及长、宽
            x, y, rw, rh = cv.boundingRect(contours[cnt])
            if p < 5 or area < 10:
                continue
            if y > (h - 50):
                continue
            # 计算最小外接矩形角度
            (x, y), (a, b), angle = cv.minAreaRect(contours[cnt]);
            angle = abs(angle)
            # 筛选标准 角度不能小于20或者大于90度或者等于90度，剔除
            if angle < 20 or angle > 160 or angle == 90.0:
                continue
            # contour的长度大于5
            if len(contours[cnt]) > 5:
            # 椭圆拟合
                (x, y), (a, b), degree = cv.fitEllipse(contours[cnt])
            # 椭圆的角度小于5 或者 角度大于160 或者角度在80和160之间，剔除
                if degree< 5 or degree>160 or 80<degree<100:
                    continue
            # 不被以上的条件剔除的，在创建的空白图像上绘制该轮廓
            cv.drawContours(out_image, contours, cnt, (255), 2, 8)
        result = self.fitLines(out_image)
        cv.imshow("contours", out_image)
        dst = cv.addWeighted(frame, 0.8, result, 0.5, 0)
        cv.imshow("lane-lines", dst)

    # 直线拟合
    def fitLines(self, image):
        h, w = image.shape
        h1 = int(h / 2 + 40)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cx = w // 2
        cy = h // 2
        left_pts = []
        right_pts = []
        for col in range(100, cx, 1):
            for row in range(cy, h, 1):
                pv = image[row, col]
                if pv == 255:
                    left_pts.append((col, row))
        for col in range(cx, w-20, 1):
            for row in range(cy, h, 1):
                pv = image[row, col]
                if pv == 255:
                    right_pts.append((col, row))
        # 检测出的左车道线数量大于2
        if len(left_pts) >= 2:
            [vx, vy, x, y] = cv.fitLine(np.array(left_pts), cv.DIST_L1, 0, 0.01, 0.01)
            y1 = int((-x * vy / vx) + y)
            y2 = int(((w - x) * vy / vx) + y)
            dy = y2 - y1
            dx = w - 1
            k = dy/dx
            c = y1

            w1 = (h1 -c)/k
            w2 = (h - c) / k
            cv.line(out, (int(w1), int(h1)), (int(w2), int(h)), (0, 0, 255), 8, 8, 0)
            self.left_line['x1'] = int(w1)
            self.left_line['y1'] = int(h1)
            self.left_line['x2'] = int(w2)
            self.left_line['y2'] = int(h)
        # 检测出的左车道线数量为1
        else:
            x1 = self.left_line['x1']
            y1 = self.left_line['y1']
            x2 = self.left_line['x2']
            y2 = self.left_line['y2']
            cv.line(out, (x1, y1), (x2, y2), (0, 0, 255), 8, 8, 0)
        # 检测出的右车道线数量大于2
        if len(right_pts) >= 2:
            x1, y1 = right_pts[0]
            x2, y2 = right_pts[len(right_pts) - 1]
            dy = y2 - y1
            dx = x2 - x1
            k = dy / dx
            c = y1 - k * x1
            w1 = (h1 - c) / k
            w2 = (h - c)/k
            cv.line(out, (int(w1), int(h1)), (int(w2), int(h)), (0, 0, 255), 8, 8, 0)
            self.right_line['x1'] = int(w1)
            self.right_line['y1'] = int(h1)
            self.right_line['x2'] = int(w2)
            self.right_line['y2'] = int(h)
        # 检测出的右车道线数量为1
        else:
            x1 = self.right_line['x1']
            y1 = self.right_line['y1']
            x2 = self.right_line['x2']
            y2 = self.right_line['y2']
            cv.line(out, (x1, y1), (x2, y2), (0, 0, 255), 8, 8, 0)
        return out


def run(path):
    img = cv.imread(path)
    detector = LaneLineDetection()
    cv.imshow("video-input", img)
    detector.process(img, 0)
    cv.waitKey(1)


if __name__ == "__main__":
    run('line.jpeg')
    cv.waitKey(0)
    cv.destroyAllWindows()