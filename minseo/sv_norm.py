import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()

def sv_norm(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)
    h=h/255
    s = (scalar.fit_transform(s)*255).astype(np.float64)
    v = (scalar.fit_transform(v)*255).astype(np.float64)
    res = cv2.merge((h,s,v)).astype(np.uint8)
    res = cv2.cvtColor(res,cv2.COLOR_HSV2RGB)
    return res