"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""
import cv2
#from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here

    H, W= img.shape
    C = 1

    K_size = 3
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float64)
    out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float64)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.median(tmp[y:y + K_size, x:x + K_size])

    denoise_img = out[pad:pad + H, pad:pad + W].astype(np.uint8)

    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    raise NotImplementedError
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here


    H, W= img.shape

    # Zero padding
    K_size = 3
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float64)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)
    tmp = out.copy()

    edge_y = out.copy()
    edge_x = out.copy()

    ## Sobel vertical
    Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    ## Sobel horizontal
    Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

    # filtering
    for y in range(H):
        for x in range(W):
            edge_y[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
            edge_x[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))



    edge_y = np.clip(edge_y, 0, 255)
    y_min = np.min(edge_y)
    y_max = np.max(edge_y)
    edge_y = 255 * (edge_y-y_min) / (y_max - y_min)


    edge_x = np.clip(edge_x, 0, 255)
    x_min = np.min(edge_x)
    x_max = np.max(edge_x)
    edge_x = 255 * (edge_x - x_min) / (x_max - x_min)


    edge_mag = np.sqrt(edge_x * edge_x + edge_y * edge_y)
    edge_mag = edge_mag.astype(np.uint8)

    edge_x = edge_x + 128
    edge_y = edge_y + 128
    edge_y = edge_y[pad: pad + H, pad: pad + W].astype(np.uint8)
    edge_x = edge_x[pad: pad + H, pad: pad + W].astype(np.uint8)

    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    H, W = img.shape

    # Zero padding
    K_size = 3
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float64)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)
    tmp = out.copy()

    edge_45 = out.copy()
    edge_135 = out.copy()

    ## Sobel vertical
    K45 = [[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]]
    ## Sobel horizontal
    K135 = [[0., -1., -2.], [1., 0., -1.], [2., 1., 0.]]

    # filtering
    for y in range(H):
        for x in range(W):
            edge_45[pad + y, pad + x] = np.sum(K45 * (tmp[y: y + K_size, x: x + K_size]))
            edge_135[pad + y, pad + x] = np.sum(K135 * (tmp[y: y + K_size, x: x + K_size]))

    edge_45 = np.clip(edge_45, 0, 255)
    edge_135 = np.clip(edge_135, 0, 255)

    edge_45 = edge_45[pad: pad + H, pad: pad + W].astype(np.uint8)
    edge_135 = edge_135[pad: pad + H, pad: pad + W].astype(np.uint8)
    #print() # print the two kernels you designed here
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = cv2.imread('task2.png', cv2.IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    cv2.imwrite('results/task2_denoise.jpg', denoise_img)
    denoise_img = cv2.imread('results/task2_denoise.jpg', cv2.IMREAD_GRAYSCALE)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    cv2.imwrite('results/task2_edge_x.jpg', edge_x_img)
    cv2.imwrite('results/task2_edge_y.jpg', edge_y_img)
    cv2.imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    cv2.imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    cv2.imwrite('results/task2_edge_diag2.jpg', edge_135_img)





