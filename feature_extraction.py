import numpy as np
import cv2
from os import listdir
from os.path import join as p_join
from os.path import abspath
from pathlib import Path 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import glob

def calculateGLCMFeatures(row, im: np.ndarray):
    """
    calculates GLCM features for a given image and stores the values in a given dictionary
    """
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(im_g, distances=[1], angles=[np.pi/2])
    for feature in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']:
        val = greycoprops(glcm, feature)[0,0]
        row[feature] = val
    return row
	
def calculateHSVFeatures(row, im):
    """
    calculates mean Hue, Saturation and Brightness for a given image and stores the values in a given dictionary
    """
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    features = ['hue', 'saturation', 'brightness']
    for i in range(len(features)):
        val = np.mean(im_hsv[...,i])
        row[features[i]] = val
    return row
    
# the bigger the sharper, the lower the blurrier
def calculateLaplacian(row, im):
    """
    calculates Laplacian (blurriness) for a given image and stores the value in a given dictionary
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    val = cv2.Laplacian(gray, -1).var()
    row['Laplacian'] = val
    return row
    
def rgb_values(row, im):
    #bgr
    blue = im[:,:,0]
    green = im[:,:,1]
    red = im[:,:,2]
    row["blue"] = np.mean(blue)
    row["green"] = np.mean(green)
    row["red"] = np.mean(red)
    return row
  
def rms_contrast(row, im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    row["contrast2"] = gray.std()
    return row
    
def calculateFeatures(row, im):
    """
    row: dictionary / data frame row
    im: image
    calculates all Features for given image and stores results in row
    """
    row = calculateGLCMFeatures(row, im)
    row = calculateHSVFeatures(row, im)
    row = calculateLaplacian(row, im)
    row = rms_contrast(row, im)
    row = rgb_values(row, im)
    return row