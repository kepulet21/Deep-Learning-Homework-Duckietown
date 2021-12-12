#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

def prep_image(image_):#preprocess one image
    img_height=60#target height
    img_width=80#target width
    #img = cv2.imread(image_)
    img = image_
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#toRGB
    img = cv2.resize(img, None,fx=(img_width/img.shape[1]), fy=(img_height/img.shape[0]), interpolation = cv2.INTER_CUBIC)#resize to target shape
    img = img[ 20:40,0:80]#crop image (get processable data)
    img = cv2.resize(img, None,fx=(1), fy=(40/20), interpolation = cv2.INTER_CUBIC)
    _ , img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)#manual threshold the image 

    lower_white = np.array([0, 0, 10])#lower filter
    upper_white = np.array([0, 0, 255])#upper filter
    hsv_ = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#hsv
    mask_ = cv2.inRange(hsv_, lower_white, upper_white)#mask (one channel)
    kernel = np.ones((3,3),np.uint8)#make 2x2 kernel, filter outliers
    mask_ = cv2.morphologyEx(mask_, cv2.MORPH_OPEN, kernel)#blue

    lower_orange = np.array([90, 0, 0])#lower filter
    upper_orange = np.array([255, 255, 255])#upper filter
    mask = cv2.inRange(hsv_, lower_orange, upper_orange)#red 
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)#red without outliers

    ret_, n = cv2.threshold(img[:,:,1],0,0,cv2.THRESH_BINARY)

    img =cv2.merge((mask, n, mask_))#merge channels to RGB image
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)#normalize the image

    return img#return preprocessed image


def prep_images(images_, image_):
    if shape(images_)[3] < 15:
        x=np.empty((40,80,3,15))
        for i in range(15):
            x[:,:,:,i] = prep_image(image_)
        return x
    else:
        images = images_
        images = np.roll(images_,-1)
        images[-1] = prep_image(image_)
        return images
        
            

