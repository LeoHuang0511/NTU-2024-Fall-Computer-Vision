import numpy as np
from PIL import Image
import os
import cv2

ori_img = cv2.imread("./lena.bmp",cv2.IMREAD_GRAYSCALE)
try:
    os.mkdir("./output")
    cv2.imwrite('./output/ori.png',ori_img)

except:
    cv2.imwrite('./output/ori.png',ori_img)

binary_img = np.zeros(ori_img.shape, dtype=int)
binary_img[ori_img > 127] = 255

downsampled_img = np.zeros((64, 64), dtype=int)
for i in range(64):
    for j in range(64):
        downsampled_img[i, j] = binary_img[8 * i, 8 * j]





def yokoi_transform(img):
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return "q"
        if b == c and (d == b and e == b):
            return "r"
        return "s"

    def yokoi_number(img, i, j):
        x0 = img[i, j]
        x1 = img[i, j + 1] if j + 1 < img.shape[1] else 0
        x2 = img[i - 1, j] if i - 1 >= 0 else 0
        x3 = img[i, j - 1] if j - 1 >= 0 else 0
        x4 = img[i + 1, j] if i + 1 < img.shape[0] else 0
        x5 = img[i + 1, j + 1] if i + 1 < img.shape[0] and j + 1 < img.shape[1] else 0
        x6 = img[i - 1, j + 1] if i - 1 >= 0 and j + 1 < img.shape[1] else 0
        x7 = img[i - 1, j - 1] if i - 1 >= 0 and j - 1 >= 0 else 0
        x8 = img[i + 1, j - 1] if i + 1 < img.shape[0] and j - 1 >= 0 else 0

        a1 = h(x0, x1, x6, x2)
        a2 = h(x0, x2, x7, x3)
        a3 = h(x0, x3, x8, x4)
        a4 = h(x0, x4, x5, x1)

        if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
            return 5
        else:
            return sum([a == 'q' for a in [a1, a2, a3, a4]])
        
    output_matrix = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:  
                output_matrix[i, j] = yokoi_number(img, i, j)
    return output_matrix

def mark_pair_relationship(img, yokoi_img):
    def pair_relationship(yokoi_img, i, j):
        if yokoi_img[i, j] != 1:
            return 2
        has_p_neighbor = any(
            0 <= ni < yokoi_img.shape[0] and 0 <= nj < yokoi_img.shape[1] and yokoi_img[ni, nj] == 1
            for ni, nj in [(i, j + 1), (i - 1, j), (i, j - 1), (i + 1, j)]
        )
        return 1 if has_p_neighbor else 2

    output = np.zeros_like(yokoi_img, dtype=int)
    for i in range(yokoi_img.shape[0]):
        for j in range(yokoi_img.shape[1]):
            if img[i, j] > 0:
                output[i, j] = pair_relationship(yokoi_img, i, j)
    return output

import numpy as np

def connected_shrink_operator(img, img_pair):
    def h_cs(b, c, d, e):
        return 1 if b == c and (d != b or e != b) else 0

    def f_cs(a1, a2, a3, a4, x0):
        return 0 if sum([a1, a2, a3, a4]) == 1 else x0

    def connected_shrink(img, i, j):
        x0 = img[i, j]
        x1 = img[i, j + 1] if j + 1 < img.shape[1] else 0  # right
        x2 = img[i - 1, j] if i - 1 >= 0 else 0                # top
        x3 = img[i, j - 1] if j - 1 >= 0 else 0                # left
        x4 = img[i + 1, j] if i + 1 < img.shape[0] else 0  # bottom
        x5 = img[i + 1, j + 1] if i + 1 < img.shape[0] and j + 1 < img.shape[1] else 0  # bottom-right
        x6 = img[i - 1, j + 1] if i - 1 >= 0 and j + 1 < img.shape[1] else 0                # top-right
        x7 = img[i - 1, j - 1] if i - 1 >= 0 and j - 1 >= 0 else 0                              # top-left
        x8 = img[i + 1, j - 1] if i + 1 < img.shape[0] and j - 1 >= 0 else 0                # bottom-left

        a1 = h_cs(x0, x1, x6, x2)
        a2 = h_cs(x0, x2, x7, x3)
        a3 = h_cs(x0, x3, x8, x4)
        a4 = h_cs(x0, x4, x5, x1)

        return f_cs(a1, a2, a3, a4, x0)

    output_img = img.copy()
    for i in range(output_img.shape[0]):
        for j in range(output_img.shape[1]):
            if output_img[i, j] > 0 and img_pair[i, j] == 1:
                output_img[i, j] = connected_shrink(output_img, i, j)

    return output_img


iter = 0
old_img = np.empty_like(downsampled_img)
while (downsampled_img == old_img).sum() != downsampled_img.shape[0] * downsampled_img.shape[1]:
    cv2.imwrite(f'./output/thinning{iter}.png',downsampled_img)
    old_img = downsampled_img.copy()
    yokoi_matrix = yokoi_transform(old_img)
    pair_img = mark_pair_relationship(old_img, yokoi_matrix)
    downsampled_img = connected_shrink_operator(old_img, pair_img)
    iter += 1

cv2.imwrite(f'./output/thinning{iter}.png',downsampled_img)





