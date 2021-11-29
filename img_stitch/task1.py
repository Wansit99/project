"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
from GetHomography import Homography

# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

def rightshift(a, b, pts1, pts2, Find_Homography):

    H = Find_Homography.GetHomography(pts2, pts1)
    H = H / H[2][2]
    points0 = np.array(
        [[0, 0], [0, a.shape[0]], [a.shape[1], a.shape[0]], [a.shape[1], 0]],
        dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, b.shape[0]], [b.shape[1], b.shape[0]], [b.shape[1], 0]],
        dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))

    points2 = cv2.perspectiveTransform(points1, H)
    points = np.concatenate((points0, points2), axis=0)

    [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(np.int32)
    [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(np.int32)

    h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    output_img = cv2.warpPerspective(b, h_translation.dot(H),
                                     (x_max - x_min, y_max - y_min))
    output_img[-y_min:a.shape[0] - y_min, -x_min:a.shape[1] - x_min] = a
    return output_img







def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat[0]
        img2_idx = mat[1]

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        a = np.random.randint(0,256)
        b = np.random.randint(0,256)
        c = np.random.randint(0,256)

        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (a, b, c), 1)      #画圆，cv2.circle()参考官方文档
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (a, b, c), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (a, b, c), 1, lineType=cv2.LINE_8, shift=0)  #画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out



def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def cal_2norm(x, y, k=2):
    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    data, index = topk(distances, 2, axis=1)
    return data, index


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(left_img, None)
    kp2, des2 = sift.detectAndCompute(right_img, None)

    data, index = cal_2norm(des1, des2, 2)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, 2)
    # # Apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])
    good = []
    k = 0.05
    for i in range(data.shape[0]):
        if k * data[i][0] > data[i][1]:
            good.append((i,index[i][1]))

    # check pairs
    img = drawMatches(left_img, kp1, right_img, kp2, good)
    # cv2.imshow('test',img)
    # cv2.waitKey(0)

    pts1 = []
    pts2 = []

    good = good[:100]
    for mat in good:
        img1_idx = mat[0]
        img2_idx = mat[1]
        pts1.append(kp1[img1_idx].pt)
        pts2.append(kp2[img2_idx].pt)

    Find_Homography = Homography()
    # Homo, in1, in2 = Find_Homography.GetHomography(pts2, pts1)
    #
    # processed = cv2.warpPerspective(right_img, Homo,(right_img.shape[1], right_img.shape[0]))
    #
    # # TO DO: implement your solution here

    result = rightshift(left_img, right_img, pts1, pts2, Find_Homography)

    # cv2.imshow('test',result)
    # cv2.waitKey(0)

    return result
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


