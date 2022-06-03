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


import scipy.interpolate as interpolate
from scipy import optimize


def sharpen_from_kernel_center(center, im):
  # creates a sharpening kernel with the given center magnitude that defines the degree of sharpening/blurring
  other = -(center-1)/8
  kernel = np.array([[other,other,other], [other,center,other], [other,other,other]])
  img = cv2.filter2D(im, -1, kernel)
  return img
  

def change_laplacian(image_path, target_value, plot=False):
    """
    reads image, manipulates sharpness and writes again to temporary file
    """
    # do not change if target is 0 bc then the feature is not relevant
    if target_value == 0:
        return
    # read the (temporary) image that is to be manipulated
    im = cv2.imread(image_path)
      
    # calculate the current laplacian value
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, -1, ksize=1)
    before = lap.var()
  
    # the amount of change in the laplacian depends on the kernel magnitude (-> the value at the center of the kernel)
    # we can model the laplacian as a function of this center value.
    # this function is mostly quadratic -> compute sharpness values for a few center values and interpolate with quadratic/cubic function
    # then find the center value that corresponds to the desired sharpness according to the interpolation
    vars = []
    # use 5 values for the interpolation, the given space has proven to work well for most transformations
    interpolation_space = np.array([0.5,1.0,1.5,3,7,8]) #np.linspace(0,6,5)
    for c in interpolation_space:
        # sharpen/blur (center<1?) image with kernel with given center. 
        img = sharpen_from_kernel_center(c, im)
        # compute laplacian of modified image
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = cv2.Laplacian(grey, -1, ksize=1).var()
        vars += [v]
    # interpolate (center, laplacian) pairs
    interp_fn = interpolate.interp1d(interpolation_space,vars,kind='quadratic', fill_value='extrapolate')
    # subtract target_value so that the 0 crossing corresponds to the desired value
    interp_fn2 = lambda x: (interp_fn(x)-target_value)
  
    if plot:
        print("before", before)
        print("desired", target_value)
        # just for plotting
        vars = []
        x_ = np.linspace(0,10,30)
        for c in x_:
            img = sharpen_from_kernel_center(c, im)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            v = cv2.Laplacian(grey, -1, ksize=1).var()
            vars += [v]
    
    x = 0
    # if the desired laplacian is not reachable, use the smallest possible
    if target_value <= interp_fn(0):
        center = 0
    # else find the kernel center that comes closest to the desired laplacian by finding 0-crossing
    else :
        try:
            center = optimize.newton(interp_fn2, 3, tol=10**(-10))
        # sometimes the desired sharpness is not reachable for an image bc local maximum is below the target, set to a predefined high value for the center then
        except:
            center = 9
            
    # sharpen image with found center
    img = sharpen_from_kernel_center(center, im)

    if plot:
        # recalculate the laplacian for the kernel that we found:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(grey, -1, ksize=1).var()
      
        print("after", var,"\n")
        plt.scatter(x_, vars)
        plt.scatter([center], [var])
        plt.plot(x_, interp_fn(x_))
        plt.show()

    # write to temp file
    cv2.imwrite(image_path, img)
    return x

def change_bgr(image_path, target_values):
    """
    reads image, manipulates bgr channels to target_values and writes back to file
    """
    # in BGR format
    im = cv2.imread(image_path)
    # current BGR values of image
    BGR = np.mean(np.mean(im, axis = 1), axis=0)
    difference = target_values - BGR
    for i in range(3):
        # do not change if target value is 0 bc that means that the feature is not relevant
        if target_values[i] == 0:
            continue
        # difference image for cv2.subtract/add
        diff = np.zeros_like(im)
        # fill with the magnitude of the difference between the means
        diff[:,:,i] = abs(difference[i])
        # if difference is negative we need to decrease the mean of the channel i
        if difference[i] < 0:
            im = cv2.subtract(im, diff)
        # if positive we need to increase it
        else:
            im = cv2.add(im, diff)
    # write the resulting image back to the file
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, im)
    return im
    
def change_hsv(image_path, target_values, change_hue=False):
    """
    reads image, manipulates HSV channels and writes back to file
    """
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # current HSV values of image
    HSV = np.mean(np.mean(im, axis = 1), axis=0)
    difference = target_values - HSV
  
    # we do not always want to modify the hue because the overlapping with RGB is very strong and hue is angular so the mean of hues is not very meaningful
    # we only change the hue if RGB are not changed
    if change_hue:
        start = 0
    else: 
        start = 1

    for i in range(start, 3):
        # do not change if target value is 0 bc then the feature is not relevant
        if target_values[i]==0:
            continue
        # difference image for cv2.subtract/add
        diff = np.zeros_like(im)
        # fill with the magnitude of the difference between the means
        diff[:,:,i] = abs(difference[i])
        # if difference is negative we need to decrease the mean of the channel i
        if difference[i] < 0:
            im = cv2.subtract(im, diff)
        # if positive we need to increase it
        else:
            im = cv2.add(im, diff)
    # write the resulting image back to the file
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    cv2.imwrite(image_path, im)
    return im
    
    
