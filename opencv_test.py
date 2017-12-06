import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#IMAGE
stop_folder = 'Traffic Sign Data Set/STOP/'
yield_folder = 'Traffic Sign Data Set/YIELD/'
donotenter_folder = 'Traffic Sign Data Set/DO NOT ENTER/'
nlf_folder = 'Traffic Sign Data Set/NO LEFT TURN/'
speedlimit_folder = 'Traffic Sign Data Set/SPEED LIMIT 25/'
oneway_folder = 'Traffic Sign Data Set/ONE WAY/'

stop_file = 'Stop_Sign_'
yield_file = 'Yield_Sign_'
donotenter_file = 'Do_Not_Enter_Sign_'
nlf_file = 'No_Left_Turn_Sign_'
speedlimit_file = 'Speed_Limit_25MPH_Sign_'
oneway_file = 'One_Way_Sign_'

png = '.png'

folder = yield_folder
filename = yield_file+'7'+png
img = cv2.imread(folder+filename)

# --opencv
# cv2.imshow('Stop Sign', img)
# cv2.waitKey(0)

# --matplotlib
# plt.imshow(img)
# plt.show()

# VIDEO CAPTURE
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Me', frame)
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# INVERSE COLORS
# cv2.imshow('Stop Sign', img)
# ret, mask = cv2.threshold(img, 120, 255,cv2.THRESH_BINARY)
# cv2.imshow('Mask', mask)
# inv_mask = cv2.bitwise_not(mask)
# cv2.imshow('Inverse Mask', inv_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# SMOOTHING/BLUR
# cv2.imshow('Stop Sign', img)
# ret, mask = cv2.threshold(img, 120, 255,cv2.THRESH_BINARY)
# cv2.imshow('Mask', mask)
# kernel = np.ones((15,15),np.float32)/225
# smoothed = cv2.filter2D(mask,-1,kernel)
# cv2.imshow('smoothed', smoothed)
# blur = cv2.GaussianBlur(mask,(15,15),0)
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# EDGE DETECTION
cv2.imshow('Stop Sign', img)
ret, mask = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
cv2.imshow('Mask', mask)
edges = cv2.Canny(mask, 100, 200)
cv2.imshow('edges', edges)
og_edge = cv2.Canny(img, 100, 200)
cv2.imshow('OG EDGES', og_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()