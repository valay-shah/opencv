# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:23:02 2019

@author: Valay
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.misc import imread, imresize, imsave, imshow
from skimage.color import rgb2gray
import os
import sys
from datetime import datetime
import imutils

#from scipy.sparse.linalg import LinearOperator
#import sklearn
def find_vector_set(diff_image, new_size):
   
    i = 0
    j = 0
    print(new_size)
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    print("vector_set",vector_set)

    print('\nvector_set shape',vector_set.shape)
    
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                #print("printing",i,j,k,block.shape)
                #print(block.shape)
                feature = block.ravel()
                #print("feature",feature)
                vector_set[i, :] = feature
                #print("vector set",vector_set)
                k = k + 5
            j = j + 5
        i = i + 1
        
    print("vector set",vector_set)        
    mean_vec   = np.mean(vector_set, axis = 0)    
    print("mean_vec",mean_vec)          
    vector_set = vector_set - mean_vec
    print("vector_set",vector_set)
    
    return vector_set, mean_vec
    
  
def find_FVS(EVS, diff_image, mean_vec, new):
    
    i = 2 
    feature_vector_set = []
    
    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
    print(np.array(feature_vector_set).shape)    
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size",FVS.shape)
    return FVS

def clustering(FVS, components, new):
    
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)

    least_index = min(count, key = count.get) 
    print("least_index",least_index)           
    print(new[0],new[1])
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    
    return least_index, change_map

   
def find_PCAKmeans(imagepath1, imagepath2):
    
    print('Operating')
    
    image_rgb_1 = imread(imagepath1)
    print("original image shape",image_rgb_1.shape)
    image1=rgb2gray(image_rgb_1)
    print("image1",image1)
    imsave("rgb_to_gray.jpg",image1)
    #image1=imread(imagepath1)
    #image1_shape = image1.shape
    
    print("image 1 shape",image1.shape)
    #print("image1",image1[:,:,0])
    #print("image1",image1[:,:,1])
    #print("image1",image1[:,:,2])
    #image1_0= 0.2989 *image1[:,:,0] 
    #print("checking",image1_0)
    image_rgb_2 = imread(imagepath2)
    print("original image shape",image_rgb_2.shape)
    image2=rgb2gray(image_rgb_2)
    
    #image2=imread(imagepath2)
    #print("image2 shape",image2.shape)
    #print(image1.shape,image2.shape) 
    #print("original new size",new_size)
    new_size = np.asarray(image1.shape) / 5
    print("new size",new_size)
    new_size = new_size.astype(int) * 5
    print("new size 2",new_size)
    image1 = imresize(image1, (new_size)).astype(np.int16)
    image2 = imresize(image2, (new_size)).astype(np.int16)
    
    diff_image = abs(image1 - image2)   
    print("diff image",diff_image)
    """for i in diff_image[0].length:
        for j in diff_image[1].length:
            if diff_image[i][j]>0:
                print("hello")"""
        
    imsave('diff.jpg', diff_image)
    print('\nBoth images resized to ',new_size)
        
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
    print("evs",EVS.shape)
        
    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    
    print('\ncomputing k means')
    
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
    
    
    
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    
    print("printing nonzero values",np.count_nonzero(np.array(change_map)))
    print(change_map.shape)
    
    #print("printing change map",change_map)
    
    
    
    change_map = change_map.astype(np.uint8)
    
    """thresh = cv2.threshold(change_map, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(image1, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.rectangle(image2, (x,y), (x+w,y+h), (0,0,255),2)
    
    imshow(image1)
    imshow(image2)"""
    
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    now=datetime.now()
    dirName = 'E:/mcte_internship/output'
    try:
        if not os.path.exists('E:/mcte_internship/output'):
            os.mkdir('E:/mcte_internship/output')
    except:
        print()
    imsave(dirName + "/" + str(now.strftime("%Y%m%d-%H%M%S")) + "changemap.jpg", change_map)
    imsave(dirName + "/" + str(now.strftime("%Y%m%d-%H%M%S")) + "cleanchangemap.jpg", cleanChangeMap)

    
if __name__ == "__main__":
    a = 'ElephantButte_08201991.jpg'
    b = 'ElephantButte_08272011.jpg'
    a1 = 'E:\mcte_internship\Change-Detection-in-Satellite-Imagery-master\Dubai_11122012.jpg'
    b1 = 'E:\mcte_internship\Change-Detection-in-Satellite-Imagery-master\Dubai_11272000.jpg'
    a2 = 'E:\mcte_internship\Change-Detection-in-Satellite-Imagery-master\Andasol_09051987.jpg'
    b2 = 'E:\mcte_internship\Change-Detection-in-Satellite-Imagery-master\Andasol_09122013.jpg'
    a3 = 'E:\mcte_internship\\cub_station_1.jpg'
    b3 = 'E:\mcte_internship\\cub_station_2.jpg'
    a4 = "E:\mcte_internship\Maharashtra\Mumbai\City\GOI\\20191230-132316map.jpg"
    b4 = "E:\mcte_internship\Maharashtra\Mumbai\City\GOI\\map.jpg"
    a5 = "E:\mcte_internship\IMG_20191231_173033.jpg"
    b5 = "E:\mcte_internship\IMG_20191231_173036.jpg"
    find_PCAKmeans(a1,b1)    