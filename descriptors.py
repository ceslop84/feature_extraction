import numpy as np
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy

img = cv2.imread('img/855215.jpg')
img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Hu Moments.
desc = cv2.HuMoments(cv2.moments(img_gray)).flatten()

#sift
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img_gray,None)
img_kp = cv2.drawKeypoints(img_gray,keypoints_1,img)
plt.imshow(img_kp)
plt.show()

# Corner Harris
img_gray_32 = np.float32(img_gray)
dst = cv2.cornerHarris(img_gray_32,2,3,0.04)
dst = cv2.dilate(dst,None) #result is dilated for marking the corners, not important
img_ch = deepcopy(img)
img_ch[dst>0.01*dst.max()]=[0,0,255] # Threshold for an optimal value, it may vary depending on the image.
plt.imshow(img_ch)
plt.show()

# Corners
corners = cv2.goodFeaturesToTrack(img_gray,25,0.01,10)
corners = np.int0(corners)
img_c = deepcopy(img)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img_c,(x,y),3,255,-1)
plt.imshow(img)
plt.show()
