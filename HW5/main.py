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
                maximum = 0
                idx = []
                jdx = []
                for (k, l) in kernel:
                    i_d = i + k
                    j_d = j + l
                    if i_d >= 0 and j_d >= 0 \
                        and i_d <= (img.shape[0]-1) and j_d <= (img.shape[1]-1):
                        maximum = max(maximum, img[i_d][j_d])
                        idx.append(i_d)
                        jdx.append(j_d)
                img_temp[(idx, jdx)] = maximum

    
    return img_temp

img_dilation = dilation(img_ori, kernel)
cv2.imwrite('./output/dilation.png',img_dilation)

# erosion
def erosion(img, kernel):
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            draw = True
            minimum = 255
            idx = []
            jdx = []
            for (k, l) in kernel:
                i_d = i + k
                j_d = j + l
                if i_d < 0 or j_d < 0 \
                    or i_d > (img.shape[0]-1) or j_d > (img.shape[1]-1) \
                    or img[i_d][j_d] <= 0:
                    draw = False
                    break
                else:
                    idx.append(i)
                    jdx.append(j)
                minimum = min(minimum, img[i_d][j_d])
            if draw:
                img_temp[(idx, jdx)] = minimum
    
    return img_temp

img_erosion = erosion(img_ori, kernel)
cv2.imwrite('./output/erosion.png',img_erosion)

# opening
def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

img_opening = opening(img_ori, kernel)
cv2.imwrite('./output/opening.png',img_opening)

# closing
def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

img_closing = closing(img_ori, kernel)
cv2.imwrite('./output/closing.png',img_closing)

