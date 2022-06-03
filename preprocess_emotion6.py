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


def get_split_from_folder(path, groundtruth):
    """
    create df with the ground truth of only the images in the given path
    the path should contain one folder for each emotion
    """
    df = pd.DataFrame()
    # for each folder 
    folders = listdir(path)[:-1]
    # for each folder:
    for i, f in enumerate(folders):
        print(f)
        try:
            l = listdir(p_join(path, f))
        except:
            print(f, "not a folder")
            continue
        # for each image:
        for file in l:
            # image id (name of image in front of .jpg)
            image_id = int(file.split(".")[0])
            row = groundtruth[(groundtruth["folder"]==f) & (groundtruth["image"]==image_id)]
            df = df.append(row, ignore_index=True)
    return df
    
# go through data frame and extract only those with "train" as split
def dataframe_from_model(groundtruth, model_split, split_class="train"):
    """
    this function returns a data frame with the groundtruth for only those images that are in the model_split dataframe and have the given split_class

    groundtruth: intial ground truth data frame of emotion6
    model_split: Anmols dataframe that contains images > 0.5 used for the classification model the label "split" that defines which image was used for training, testing, validation
    split_class: which set to extract (train, test, val)
    """
    emotion_distr_model = pd.DataFrame()  
    for i in range(len(groundtruth)):
        row = groundtruth.loc[i]
        folder = row["folder"]
        image = row["image"]
        if (folder, image) not in zip(model_split["folder"].values, model_split["image"].values):
            continue
        split = model_split[(model_split["folder"]==folder) & (model_split['image'] == image)].split.values[0]
        if split != split_class:
            continue
        row_new = row
        emotion_distr_model = emotion_distr_model.append(row_new, ignore_index=True)
    return emotion_distr_model
    
def threshold_images(groundtruth, threshold = 0.5, greater_than = True):
    """
    this function extracts the groundtruth for only those images are below or over a given threshold
    
    groundtruth: intial ground truth data frame of emotion6
    threshold: threshold for maximum emotion probability
    greater_than: if True use only images with max_prob > threshold
        if False use only images with max_prob < threshold
    """
    emotion_distr_model = pd.DataFrame()  
    emotions = list(np.unique(groundtruth.folder.values)) + ["neutral"]
    print(emotions)

    for i in range(len(groundtruth)):
        row = groundtruth.loc[i]
        probabilities = row.values[4:] # ground truth probabilities
        # compute most likely emotion
        label = emotions[np.argmax(probabilities)]
        # only if prob >= threshold
        if greater_than and max(probabilities) <= threshold:
            continue
        elif not greater_than and max(probabilities) >= threshold:
            continue
        row_new = row
        emotion_distr_model = emotion_distr_model.append(row_new, ignore_index=True)
    return emotion_distr_model