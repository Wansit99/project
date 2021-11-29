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
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random

# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

k = 2
n = 0.5


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    # raise NotImplementedError
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_img, None)
    kp2, des2 = sift.detectAndCompute(right_img, None)
    kp1 = get_keypoints(kp1)
    kp2 = get_keypoints(kp2)

    good_matches1, good_matches2 = knn(kp1, des1, kp2, des2, k)
    homography = RANSAC(good_matches1, good_matches2)
    #
    h, w, c = left_img.shape
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(4, 1, 2)
    corners = cv2.perspectiveTransform(corners, homography)
    corners = corners.reshape(4, 2)
    h2, w2, c2 = right_img.shape
    corners2 = np.float32([[0, 0], [0, h2], [w, h2], [w2, 0]])
    corners = np.concatenate((corners, corners2), axis=0)
    x_min = int(np.ceil(min(corners, key=lambda x: x[0])[0]))
    x_max = int(np.ceil(max(corners, key=lambda x: x[0])[0]))
    y_min = int(np.ceil(min(corners, key=lambda x: x[1])[1]))
    y_max = int(np.ceil(max(corners, key=lambda x: x[1])[1]))
    #
    final_shape = (x_max-x_min, y_max-y_min)
    #
    translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    result_img = cv2.warpPerspective(left_img, np.dot(translate, homography), final_shape)
    result_img[-y_min:(h2 + -y_min), -x_min:(w2 + -x_min)] = right_img

    return result_img


def get_keypoints(kp):
    kp_list = []
    for x in kp:
        kp_list.append([x.pt[0], x.pt[1]])
    return kp_list


def knn_helper(des1, kp2, des2, num):
    matched_list = {}
    count1 = 0
    for x in des1:
        x_vector = np.asarray(x)
        y_matrix = np.asarray(des2)
        dist_list = np.linalg.norm(x_vector - y_matrix, axis=1)
        dist_list = [[index, item] for index, item in enumerate(dist_list)]
        dist_list.sort(key=lambda v: v[1])
        dist_list = dist_list[:num]
        if str(count1) not in matched_list:
            matched_list[str(count1)] = []
        matched_list[str(count1)].append([kp2[dist_list[0][0]], dist_list[0][1]])
        matched_list[str(count1)].append([kp2[dist_list[1][0]], dist_list[1][1]])
        count1 += 1
    return matched_list


def knn(kp1, des1, kp2, des2, num):
    matched_list = knn_helper(des1, kp2, des2, num)
    matched_list2 = knn_helper(des2, kp1, des1, num)
    good_matches1, good_matches2 = remove_ambiguous_matches(matched_list, kp1, matched_list2, kp2)

    return good_matches1, good_matches2


def ratio_test(matched_list):
    for key, val in list(matched_list.items()):
        if val[0][1] >= val[1][1] * n:
            del matched_list[key]
        else:
            matched_list[key][0].pop(1)
            matched_list[key].pop(1)

    return matched_list


def remove_ambiguous_matches(matched_list1, kp1, matched_list2, kp2):
    good_matches1 = []
    good_matches2 = []
    matched_list1 = ratio_test(matched_list1)
    matched_list2 = ratio_test(matched_list2)

    for key1, val1 in matched_list1.items():
        for key2, val2 in matched_list2.items():
            if (kp1[int(key1)] in val2[0]) and (kp2[int(key2)] in val1[0]):
                good_matches1.append(kp1[int(key1)])

    for key1, val1 in matched_list2.items():
        for key2, val2 in matched_list1.items():
            if (kp2[int(key1)] in val2[0]) and (kp1[int(key2)] in val1[0]):
                good_matches2.append(kp2[int(key1)])
    return good_matches1, good_matches2


def RANSAC(goodmatch1, goodmatch2):
    n = 4
    k = 5000
    t = 5
    max_inliers = []
    goodmatch1_copy = np.asarray(goodmatch1)
    goodmatch2_copy = np.asarray(goodmatch2)
    for i in range(k):
        random_four = np.random.randint(len(goodmatch1), size=n)
        H = get_homography(goodmatch1_copy[random_four], goodmatch2_copy[random_four])
        inliers = []
        for x in range(len(goodmatch1)):
            point = [goodmatch1[x][0], goodmatch1[x][1], 1]
            point = np.asarray(point).T
            estimate = np.dot(H, point)
            estimate = estimate / estimate[2]
            point2 = [goodmatch2[x][0], goodmatch2[x][1], 1]
            point2 = np.asarray(point2).T
            residual_distance = np.linalg.norm(point2 - estimate)
            if residual_distance < t:
                inliers.append([goodmatch1[x], goodmatch2[x]])

        if len(max_inliers) < len(inliers):
            max_inliers = inliers
    final_match1 = []
    final_match2 = []
    for i in range(len(max_inliers)):
        final_match1.append(max_inliers[i][0])
        final_match2.append(max_inliers[i][1])
    finalH = get_homography(final_match1, final_match2)
    return finalH


def get_homography(mp_list1, mp_list2):
    A = []
    length = len(mp_list1)
    for i in range(length):
        row1 = [mp_list1[i][0], mp_list1[i][1], 1, 0, 0, 0, -mp_list2[i][0] * mp_list1[i][0],
                -mp_list2[i][0] * mp_list1[i][1], -mp_list2[i][0]]
        row2 = [0, 0, 0, mp_list1[i][0], mp_list1[i][1], 1, -mp_list2[i][1] * mp_list1[i][0],
                -mp_list2[i][1] * mp_list1[i][1], -mp_list2[i][1]]
        A.append(row1)
        A.append(row2)

    u, s, vh = np.linalg.svd(np.asarray(A))
    H = np.reshape(vh[8], (3, 3))
    H = H/H[2][2]

    return H


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)

