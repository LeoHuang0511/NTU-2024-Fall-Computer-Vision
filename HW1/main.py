import cv2
import numpy as np

#read img
img = cv2.imread("./lena.bmp")
cv2.imwrite('./output/ori_img.jpg',img)

#up-side-down
ud_img = np.empty(img.shape)
for i in range(img.shape[0]):
    ud_img[img.shape[0]-i-1,:] = img[i,:]
cv2.imwrite('./output/updown_img.jpg',ud_img)

#left-side-right
rl_img = np.empty(img.shape)
for i in range(img.shape[1]):
    rl_img[:,img.shape[1]-i-1] = img[:,i]
cv2.imwrite('./output/rightleft_img.jpg',rl_img)

#diagnonally flip
diag_img = np.empty(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        diag_img[img.shape[0]-i-1,img.shape[1]-j-1] = img[i,j]
cv2.imwrite('./output/diagflip_img.jpg',diag_img)

#rotate
(h, w) = img.shape[:2]
    
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)  

cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

rot_img = cv2.warpAffine(img, M, (new_w, new_h))
cv2.imwrite('./output/rotated_img.jpg',rot_img)

#shrink
resize_img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
shrinked_img = np.zeros(img.shape)
shrinked_img[:resize_img.shape[0],:resize_img.shape[1]] = resize_img
cv2.imwrite('./output/shrinked_img.jpg',shrinked_img)

#binarize
binary_img = img.copy()
binary_img[np.where(img>128)] = 255
binary_img[np.where(img<=128)] = 0
cv2.imwrite('./output/binarized_img.jpg',binary_img)