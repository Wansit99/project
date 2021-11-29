###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners



import cv2 as cv
# def show_point(imgname):
    
    
#     img=imread(imgname)
#     # img=cvtColor(img,COLOR_BGR2GRAY)
#     # print(img.shape)
#     # cv2.imshow('name',img)
#     size=(4,9)
#     ok,corners=findChessboardCorners(img,size,None)

#     print(corners.shape)
#     if ok:
#         for pt in corners:
#             point=pt[0]
#             cv.circle(img,center=(int(point[0]),int(point[1])),radius=10,color=(0,0,255),thickness=-1)
#         cv.imshow('img',img)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
    
#     return None ,None

def build_matrix(world_points,pic_points):
    matrix=[]
    for (world_point,pic_point) in zip(world_points,pic_points):
        matrix.append([
            world_point[0],world_point[1],world_point[2],1,0,0,0,0,-pic_point[0][0]*world_point[0],-pic_point[0][0]*world_point[1],-pic_point[0][0]*world_point[2],-pic_point[0][0]
        ])

        matrix.append([
            0,0,0,0,world_point[0],world_point[1],world_point[2],1,-pic_point[0][1]*world_point[0],-pic_point[0][1]*world_point[1],-pic_point[0][1]*world_point[2],-pic_point[0][1]
        ])

    matrix=np.array(matrix)
    #print(matrix.shape)
    
    return matrix

def get_points(imgname):
    img=cv.imread(imgname)
    img=cvtColor(img,COLOR_BGR2GRAY)
    # world coordinate
    world_points=np.array(
        [
            [4,0,4],
            [4,0,3],
            [4,0,2],
            [4,0,1],
            
            [3,0,4],
            [3,0,3],
            [3,0,2],
            [3,0,1],

            [2,0,4],
            [2,0,3],
            [2,0,2],
            [2,0,1],

            [1,0,4],
            [1,0,3],
            [1,0,2],
            [1,0,1],

            [0,0,4],
            [0,0,3],
            [0,0,2],
            [0,0,1],

            [0,4,4],
            [0,4,3],
            [0,4,2],
            [0,4,1],
            
            [0,3,4],
            [0,3,3],
            [0,3,2],
            [0,3,1],

            [0,2,4],
            [0,2,3],
            [0,2,2],
            [0,2,1],

            [0,1,4],
            [0,1,3],
            [0,1,2],
            [0,1,1],
        ]
    )*10

    size=(4,9)#to find the 36 points(plus 4 points to confirm the position)
    ok,corners=findChessboardCorners(img,size,None)

    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    img=drawChessboardCorners(img, size, corners, ok)#draw the points，
    #cv.imshow('img',img)#steps only for checking
    #cv.imwrite('ttt.png',img)
    #cv.waitKey(0)
    corners=cornerSubPix(img, corners, (11, 11), (-1, -1), stop_criteria)


    return world_points ,corners


def calibrate(imgname):
    world_points,pic_points=get_points(imgname)
    matrix=build_matrix(world_points,pic_points)
    #print(matrix.shape)# to remember
    U,sigma,V_T=np.linalg.svd(matrix,full_matrices=False)
    w,v = np.linalg.eig(np.transpose(matrix)@matrix)
    lamda=np.min(w)
    m_mat = lamda * V_T[-1]
    #m_mat=V_T[-1][:]
    #print(matrix@m_mat)
    
    m_mat=m_mat.reshape(3,4)
    #print(m_mat)# to check

    m1=np.array([m_mat[0][0],m_mat[0][1],m_mat[0][2]])
    m2=np.array([m_mat[1][0],m_mat[1][1],m_mat[1][2]])
    m3=np.array([m_mat[2][0],m_mat[2][1],m_mat[2][2]])
    m4=np.array([m_mat[0][3],m_mat[1][3],m_mat[2][3]])

    ox=m1.T@m3
    oy=m2.T@m3
    fx=np.sqrt(m1.T@m1-ox*ox)
    fy=np.sqrt(m2.T@m2-oy*oy)

    intrinsic_params=np.array([
        [fx,fy,ox,oy],
    ])
    is_constant=True

    return intrinsic_params,is_constant



if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('left.jpg')
    # find_points('checkboard.png')
    # show_point('checkboard.png')
    print(intrinsic_params)
    print(is_constant)


