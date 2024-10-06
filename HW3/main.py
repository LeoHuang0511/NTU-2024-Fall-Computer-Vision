import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#read img
img = cv2.imread("./lena.bmp",cv2.IMREAD_GRAYSCALE)
# os.mkdir("./outputs)

def histogram(im):
    his = np.zeros(256)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            his[im[i,j]] += 1
    return his
his = histogram(img)
cv2.imwrite('./output/ori_img.png',cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR))
plt.bar(np.arange(his.shape[0]), his)
plt.savefig('./output/histogram.png')

plt.cla()


divided_img = img.copy()//3
divided_his = histogram(divided_img)
cv2.imwrite('./output/divided_img.png',cv2.cvtColor(divided_img.copy(), cv2.COLOR_GRAY2BGR))
plt.bar(np.arange(divided_his.shape[0]), divided_his)
plt.savefig('./output/divided_histogram.png')

plt.cla()

PDF = divided_his / (divided_img.shape[0]*divided_img.shape[1])
CDF = np.zeros_like(PDF)
CDF[0] = PDF[0]
for i in range(1,len(CDF)):
    CDF[i] = CDF[i-1] + PDF[i]
CDF = np.round(CDF*255)
new_PDF = np.zeros_like(PDF)
eq_img = divided_img.copy()
for idx, value in enumerate(CDF):
    new_PDF[int(value)] += PDF[idx]
    eq_img[np.where(divided_img==idx)] = value
new_PDF *= (divided_img.shape[0]*divided_img.shape[1])
cv2.imwrite('./output/eq_img.png',cv2.cvtColor(eq_img.copy(), cv2.COLOR_GRAY2BGR))
plt.bar(np.arange(new_PDF.shape[0]), new_PDF)
plt.savefig('./output/eq_histogram.png')
