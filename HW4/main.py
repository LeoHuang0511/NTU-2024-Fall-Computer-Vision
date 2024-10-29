import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#read img
img_ori = cv2.imread("./lena.bmp",cv2.IMREAD_GRAYSCALE)
try:
    os.mkdir("./output")
    cv2.imwrite('./output/ori.png',img_ori)

except:
    cv2.imwrite('./output/ori.png',img_ori)


# binarizing
def binarize(img):
    b = img.copy()
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i,j] >= 128:
                b[i,j] = 255
            else:
                b[i,j] = 0
    return b
img_bin = binarize(img_ori)
cv2.imwrite('./output/binarized.png',img_bin)


kernel = np.array([[-2, -1], [-2, 0], [-2, 1],
                   [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                   [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                   [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                   [2, -1], [2, 0], [2, 1]])


# dilation
def dilation(img, kernel):
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                for (k, l) in kernel:
                    i_d = i + k
                    j_d = j + l
                    if i_d >= 0 and j_d >= 0 \
                        and i_d <= (img.shape[0]-1) and j_d <= (img.shape[1]-1):
                        img_temp[i_d][j_d] = 255
    
    return img_temp

img_dilation = dilation(img_bin, kernel)
cv2.imwrite('./output/dilation.png',img_dilation)

# erosion
def erosion(img, kernel):
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # if img[i][j] > 0:
            draw = True
            for (k, l) in kernel:
                i_d = i + k
                j_d = j + l
                if i_d < 0 or j_d < 0 \
                    or i_d > (img.shape[0]-1) or j_d > (img.shape[1]-1) \
                    or img[i_d][j_d] <= 0:
                    draw = False
                    break
            if draw:
                img_temp[i][j] = 255
    
    return img_temp

img_erosion = erosion(img_bin, kernel)
cv2.imwrite('./output/erosion.png',img_erosion)

# opening
def opening(img, kernel):
    return dilation(erosion(img_bin, kernel), kernel)

img_opening = opening(img_bin, kernel)
cv2.imwrite('./output/opening.png',img_opening)

# closing
def closing(img, kernel):
    return erosion(dilation(img_bin, kernel), kernel)

img_closing = closing(img_bin, kernel)
cv2.imwrite('./output/closing.png',img_closing)

# hit and miss
def h_and_m(img):
    J_kernel = [[0, -1], [0, 0], [1, 0]]
    K_kernel = [[-1, 0], [-1, 1], [0, 1]]

    img_comp = -img + 255
    img_j = erosion(img, J_kernel)
    img_k = erosion(img_comp, K_kernel)
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_j[i][j] > 0 and img_k[i][j] > 0:
                img_temp[i][j] = 255

    return img_temp


img_hnm = h_and_m(img_bin)
cv2.imwrite('./output/h_and_m.png',img_hnm)






