import numpy as np 
import cv2
import pandas as pd

class k_means:
    def __init__(self,original_image):
        self.img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        self.img = cv2.normalize(self.img,None,0,255,cv2.NORM_MINMAX)
        self.vectorized = self.img.reshape((-1,3))
        self.vectorized = np.float32(self.vectorized)

    
    def process(self, K):
        criteria = (cv2.TERM_CRITERIA_EPS + 
			        cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

        attempts = 10 
        ret,label,center = cv2.kmeans(self.vectorized,K,None,criteria,attempts,
                                        cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        self.result_image = res.reshape((self.img.shape))
        return self.result_image
    
'''
from kmeans import k_means
image = cv2.imread(PATH)
k_mean_image = k_means(image).process(5) #change number to control the number of colors.
'''