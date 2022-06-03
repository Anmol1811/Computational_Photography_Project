# Computational_Photography_Project
 
This repo contains the code to extract low level features from images to create and train models for Emotion Prediction and filters to modify the images to illicit a different emotion. We also use CNN models to predict hte emotion and provide the code for training and inference 

## Installation
------------
To run the files, first add the data and models from the drive link(https://drive.google.com/drive/folders/108l2zUILNT90XD2O5vM8mnOhAGyVgDNS?usp=sharing) to your google drive as a shortcut. 
Then clone this repositiory:
```
git clone https://github.com/Anmol1811/Computational_Photography_Project.git
cd Computational_Photography_Project
```
To run the notebooks, please use colab, since the file dependencies and the trained models are of a very large size and will take very long to download.

- The following libraries are required to run this code:
Pytorch, OpenCV, Numpy, Pandas, PIL, matplotlib, glob, json, mmcv, artemis


## Dataset
------------
  
* Artemis dataset. To read more about it visit the original git(https://github.com/optas/artemis).
* Emotion6 dataset.(https://rpand002.github.io/emotion.html)

## Usage
------------
1. Helper files for extracting and transforming features and to generate some intermediate csvs.
	* feature_extraction.py: Helper file for feature extraction. 
	* feature_transform.py: Helper file for feature transformation
	* hist_matching_helpers.py: Helper file for histogram matching approach
	* preprocess_emotion6.py: Helper file for editing folder structure
2.	Correlation for low level features
	* emotion6_correlation.ipynb
	* wikiart_correlation.ipynb
	* artphoto_abstract_correlation.ipynb
3. ML and Deep learning models for Artemis and Emotion6 data: The ML model is trained using 5 fold Cross Validation and Gridsearch for finding the best params. The deep models use a resnet34 backbone with an MLP layer. Before training, in the case of artemis, you may need to generate the image histograms. Follow the instructions on git for the same if there is any error.
	* ML_classifier_Emotion6.ipynb 
	* ML_classifier_Artemis.ipynb 
	* Artemis_DeepNet.ipynb
 	* Emotion6_DeepNet.ipynb
4. Histogram matching 
	* Hist_matching_final.ipynb: Contains the code for low level transformations using Histogram Matching 
5. CSVs
	* wikiart_groundtruth.csv: Contains ground truth data for Artemis.
	* em6_groundtruth.csv: Contains ground truth data for emotion6.
	* emotion6_filtered_split.csv: Contains data for images with a dominant emotion

## Extra notebooks and experiments
--------------
* We train another ML Model based on features extracted using https://github.com/yilangpeng/computational-aesthetics.
* We trained a DL model using a CLIP backend. CLIP&Avg_Hist.ipynb. However, this is not fine tuned and does not perform as well as the Resnet34 Model.
* For other things we tried(DL Models/Filters), we share the drive link(https://drive.google.com/drive/folders/1tMW7kRVrirU9R4wxlCq58YDRba-UZ3Tr?usp=sharing). Please ask for access
