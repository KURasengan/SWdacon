import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path = "C:/SWdacon/minseo/data/"
csv_file = path+'train.csv'
train_img_path = path+'train_img/'
csv = pd.read_csv(csv_file)
img_num = 8
mask_rle = csv.iloc[img_num, 2]
image_path = csv.iloc[img_num, 1]
image = cv2.imread(path+ image_path[2:])
x,y=100,100
image = image[y:y+224,x:x+224]

hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_image)

scalar = MinMaxScaler()
scalar2 = MinMaxScaler(feature_range=(0.2,0.7))
h=h.astype(np.float64)
# h = (scalar.fit_transform(h)*255).astype(np.float64)
s = (scalar.fit_transform(s)*255).astype(np.float64)
v = (scalar.fit_transform(v)*255).astype(np.float64)

sv_image = cv2.merge((h,s,v)).astype(np.uint8)
h = (scalar.fit_transform(h)*255).astype(np.float64)
hsv_image = cv2.merge((h,s,v)).astype(np.uint8)
image_norm_sv = cv2.cvtColor(sv_image,cv2.COLOR_HSV2BGR)
image_norm_hsv = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)
b,g,r = cv2.split(image)
b = (scalar.fit_transform(b)*255).astype(np.float64)
g = (scalar.fit_transform(g)*255).astype(np.float64)
r = (scalar.fit_transform(r)*255).astype(np.float64)
image_norm_bgr = cv2.merge((b,g,r)).astype(np.uint8)

cv2.imshow('hsv norm',image_norm_hsv)
cv2.imshow('sv norm',image_norm_sv)
cv2.imshow('ori',image)
cv2.imshow('bgr norm',image_norm_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows