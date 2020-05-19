# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:25:52 2019

@author: liuyuntao
"""

# with open("决策树&RF.txt", "r", encoding="utf-8") as f:
#     cont = f.readline()
#     print(cont)
import cv2
import numpy as np 
import matplotlib.pylab as plt 
from skimage.filters import gaussian 
from skimage import morphology
from skimage import io,data 
from skimage import img_as_ubyte 
from PIL import Image 
import imutils 
import pytesseract 
import math 
import sys,os 
import copy
import requests
from PIL import Image
from io import BytesIO
import csv
import re
from scipy import signal


def getScale(img):
    (h, w, _) = img.shape
    d = max(h, w)
    return (1500.0 / d)

def adjust_gamma(imgs, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def calcAndDrawHist(image, color):  
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])
    #print(hist)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8)  
    hpt = int(0.9* 256);  
    for h in range(256):  
        intensity = int(hist[h]*hpt/maxVal)  
        cv2.line(histImg,(h,256), (h,256-intensity), color)  
    return histImg;

def gray_scale(image, minGray, maxGray):
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 0:
                continue
            if image[i ,j] < minGray:
                image[i, j] = 10
                continue
            if image[i ,j] >= maxGray:
                image[i, j] = 255
                continue
            image[i, j] = int(255/(maxGray-minGray)*(image[i,j]-minGray)+0.5)
    return image

def Hessian2D(I,Sigma):
    if Sigma<1:
        print("error: Sigma<1")
        return -1
    I=np.array(I,dtype=float)
    Sigma=np.array(Sigma,dtype=float)
    S_round=np.round(3*Sigma)

    [X,Y]= np.mgrid[-S_round:S_round+1,-S_round:S_round+1]
    #构建卷积核：高斯函数的二阶导数
    DGaussxx = 1/(2*math.pi*pow(Sigma,4)) * (X**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
    DGaussxy = 1/(2*math.pi*pow(Sigma,6)) * (X*Y) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))   
    DGaussyy = 1/(2*math.pi*pow(Sigma,4)) * (Y**2/pow(Sigma,2) - 1) * np.exp(-(X**2 + Y**2)/(2*pow(Sigma,2)))
  
    Dxx = signal.convolve2d(I,DGaussxx,boundary='fill',mode='same',fillvalue=0)
    Dxy = signal.convolve2d(I,DGaussxy,boundary='fill',mode='same',fillvalue=0)
    Dyy = signal.convolve2d(I,DGaussyy,boundary='fill',mode='same',fillvalue=0)

    return Dxx,Dxy,Dyy

def eig2image(Dxx,Dxy,Dyy):
    Dxx=np.array(Dxx,dtype=float)
    Dyy=np.array(Dyy,dtype=float)
    Dxy=np.array(Dxy,dtype=float)
    if (len(Dxx.shape)!=2):
        print("len(Dxx.shape)!=2,Dxx不是二维数组！")
        return 0

    tmp = np.sqrt( (Dxx - Dyy)**2 + 4*Dxy**2)

    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x**2 + v2y**2)
    i=np.array(mag!=0)

    v2x[i==True] = v2x[i==True]/mag[i==True]
    v2y[i==True] = v2y[i==True]/mag[i==True]

    v1x = -v2y 
    v1y = v2x

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check=abs(mu1)>abs(mu2)
            
    Lambda1=mu1.copy()
    Lambda1[check==True] = mu2[check==True]
    Lambda2=mu2
    Lambda2[check==True] = mu1[check==True]
    
    Ix=v1x
    Ix[check==True] = v2x[check==True]
    Iy=v1y
    Iy[check==True] = v2y[check==True]
    
    return Lambda1,Lambda2,Ix,Iy

def FrangiFilter2D(I):
    I=np.array(I,dtype=float)
    defaultoptions = {'FrangiScaleRange':(1,10), 'FrangiScaleRatio':2, 'FrangiBetaOne':0.5, 'FrangiBetaTwo':15, 'verbose':True,'BlackWhite':True};  
    options=defaultoptions

    sigmas=np.arange(options['FrangiScaleRange'][0],options['FrangiScaleRange'][1],options['FrangiScaleRatio'])
    sigmas.sort()#升序

    beta = 2*pow(options['FrangiBetaOne'],2)  
    c = 2*pow(options['FrangiBetaTwo'],2)
    #存储滤波后的图像
    shape=(I.shape[0],I.shape[1],len(sigmas))
    ALLfiltered=np.zeros(shape) 
    ALLangles  =np.zeros(shape) 
    #Frangi filter for all sigmas 
    Rb=0
    S2=0
    for i in range(len(sigmas)):
        #Show progress
        if(options['verbose']):
            print('Current Frangi Filter Sigma: ',sigmas[i])
        #Make 2D hessian
        [Dxx,Dxy,Dyy] = Hessian2D(I,sigmas[i])
        #Correct for scale 
        Dxx = pow(sigmas[i],2)*Dxx  
        Dxy = pow(sigmas[i],2)*Dxy  
        Dyy = pow(sigmas[i],2)*Dyy
        #Calculate (abs sorted) eigenvalues and vectors  
        [Lambda2,Lambda1,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)  
        #Compute the direction of the minor eigenvector  
        angles = np.arctan2(Ix,Iy)  
        #Compute some similarity measures  
        Lambda1[Lambda1==0] = np.spacing(1)
        
        Rb = (Lambda2/Lambda1)**2  
        S2 = Lambda1**2 + Lambda2**2
        #Compute the output image
        Ifiltered = np.exp(-Rb/beta) * (np.ones(I.shape)-np.exp(-S2/c))
        #see pp. 45  
        if(options['BlackWhite']): 
            Ifiltered[Lambda1<0]=0
        else:
            Ifiltered[Lambda1>0]=0
        #store the results in 3D matrices  
        ALLfiltered[:,:,i] = Ifiltered 
        ALLangles[:,:,i] = angles
        # Return for every pixel the value of the scale(sigma) with the maximum   
        # output pixel value  
        if len(sigmas) > 1:
            outIm=ALLfiltered.max(2)
        else:
            outIm = (outIm.transpose()).reshape(I.shape)
    return outIm

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def deleteBackground(image):
    rows, cols = image.shape
    minColor = 255
    maxColor = 10
    for i in range(rows):
        for j in range(cols):
            if image[i, j] <= 10:
                image[i, j] = 0
                continue
            if image[i, j] > maxColor:
                maxColor = image[i, j]
            if image[i, j] < minColor:
                minColor = image[i, j]
    return image, maxColor, minColor

def gray_Scale(image):
    rows, cols = image.shape
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    #print(A)
    #print(B)
    if abs(A - B) == 0:
        return image
    C = np.mean(flat_gray)
    #print('A = %d,B = %d' %(A,B))
    output = np.uint8(255 / (B - A) * (image - A) + 0.5)
    return output

def USM(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)
    # alpha 0 - 5
    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img
    img_out = img_out/255.0
    # 饱和处理
    mask_1 = img_out  < 0 
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return img_out

if __name__ == '__main__':
    image = cv2.imread('C:\\Users\\liuyuntao\\Desktop\\1.png')
    '''
    (h,w,_) = image.shape
    if max(h, w)>1500:
        scale = getScale(image)
        size = (int(w * scale), int(h * scale))  
        image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)   
    '''
    image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(image)
    
    g, maxColor, minColor = deleteBackground(g)
    #histImgG = calcAndDrawHist(g, [0, 255, 0])
    hist= cv2.calcHist([g], [0], None, [256], [0.0,255.0])
    maxN = 0
    color = 0
    for i in range(10, len(hist)):
        if maxN < int(hist[i]):
            maxN = int(hist[i])
            color = i
    #print(maxN)
    #print(color)
    #rows, cols = g.shape
    #flat_gray = g.reshape((cols * rows,)).tolist()
    minGray = int((color + minColor)/2)
    maxGray = int((color + maxColor)/2)
    
    minStep = 0.00
    minGray = int(minGray + (maxGray - minGray) * minStep)
    maxStep = 0.00
    maxGray = int(maxGray - (maxGray - minGray) * maxStep)

    g = gray_scale(g, minGray, maxGray)
    
    #g = USM(g) * 255
    #g = g.astype('uint8') 
    #histImgG = calcAndDrawHist(g, [0, 255, 0])
    
    blood = cv2.normalize(g.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
    outIm=FrangiFilter2D(blood)
    img=outIm*(10000)
    #print(img)
    img = gray_Scale(img)
    '''
    rows, cols = img.shape
    flat_gray = img.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print(A)
    print(B)
    histImgG = calcAndDrawHist(img, [0, 255, 0])
    '''
    (_, img) = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img, 7)
    
    img = morphology.remove_small_holes(img, 9)
    #contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #for i in range(len(contours)):
    #    area = cv2.contourArea(contours[i])
    #    if area < 4:
    #        cv2.drawContours(img,[contours[i]],0,0,-1)
    
    #ele = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #img = cv2.dilate(img, ele, iterations=1)
    #img = cv2.medianBlur(img, 9)
    
    plt.imshow(img)
    plt.show()