import numpy as np
import cv2
import random

class Homography:
    def __init__(self):
        self.Homography = None
        self.dist = None
    def _GetSample(self,data1,data2):
        if len(data1) != len(data2) or len(data1) != 4:
            print("error data")
            raise ValueError("Not enough input data to fit the model.")
            return None
        tmp = []
        for (x1, y1), (x2, y2) in zip(data1, data2):
            # zip can take the corresponding match point in left and right image.
            tmp1 = [x1, y1, 1,
                    0, 0, 0,
                    -x2 * x1, -x2 * y1, -x2]
            tmp2 =[0, 0, 0,
                   x1, y1, 1,
                   -y2 * x1, -y2 * y1, -y2]
            tmp.append(tmp1)
            tmp.append(tmp2)

        u, s, v = np.linalg.svd(np.asarray(tmp))
        H = np.reshape(v[8], (3, 3))
        H = H / H[2][2]

        return H
    def _RANSAC(self,vec1,vec2,min_samples,iterations=1000,eps=0.1,random_seed=44):
        random.seed(random_seed)
        if len(vec1) <= min_samples or len(vec2) <= min_samples:
            raise ValueError("Not enough input data to fit the model.")
        best_Hom = 0
        Best_Exist = False
        best_inliers = 0 
        best_iteration = None
        best_inliers1 = None
        best_inliers2 = None
        for i in range(iterations):
            Num_inliers = 0
            indices = list(range(len(vec1)))
            random.shuffle(indices)
            inLiers1 = [vec1[i] for i in indices[:min_samples]]
            inLiers2 = [vec2[i] for i in indices[:min_samples]]
            Shuffled_data1 = [vec1[i] for i in indices[min_samples:]]
            Shuffled_data2 = [vec2[i] for i in indices[min_samples:]]
            try:
                Hom = self._GetSample(inLiers1,inLiers2)
                for index in range(len(Shuffled_data1)):
                    I = (1,)
                    Projection_Point = np.array(Shuffled_data2[index])
                    # print("投影点")
                    # print(Projection_Point)
                    Original_Point =  np.array(Shuffled_data1[index]+I).reshape(3,1)
                    temp = np.dot(Hom,Original_Point)
                    Pro_p = np.array([ temp[0]/temp[2] , temp[1]/temp[2] ])
                    # print("计算点")
                    # print(Pro_p)
                    dist = np.sqrt(np.square(Pro_p[0]-Projection_Point[0])+np.square(Pro_p[1]-Projection_Point[1]))
                    # print("第%d轮第%d个匹配点的距离为%lf"%(i,index,dist))
                    if dist < eps:
                        Num_inliers += 1
                if Num_inliers > 0:
                    if Num_inliers > best_inliers:
                        best_inliers = Num_inliers
                        best_Hom = Hom
                        best_iteration = i
                        Best_Exist = True
                        best_inliers1 = inLiers1
                        best_inliers2 = inLiers2
            except ValueError as e:
                print(e)
        if Best_Exist == False:
            raise ValueError("CANNOT FIND A GOOD HOMOGRAPHY")
        else:
            return best_Hom

    def GetHomography(self,vec1,vec2):
        return self._RANSAC(vec1,vec2,4)