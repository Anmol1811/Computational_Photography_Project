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
	* feature_transform.py
	* feature_extraction.py
	* preprocess_emotion6.py
2.	Correlation for low level features
	* emotion6_correlation.ipynb
	* wikiart_correlation.ipynb
	* artphoto_abstract_correlation.ipynb
3. ML and Deep learning models for Artemis and Emotion6 data
	* ML_classifier_Emotion6.ipynb
	* ML_classifier_Artemis.ipynb
	* Artemis_DeepNet.ipynb
	* Emotion6_DeepNet.ipynb
4. Histogram matching 
	Hist_matching_final.ipynb
5. CSVs
	* wikiart_groundtruth.csv: Contains ground truth data for Artemis.
	* em6_groundtruth.csv: Contains ground truth data for emotion6.
	* emotion6_filtered_split.csv: Contains data for images with a dominant emotion
For other things we tried(DL Models/Filters), we share the drive link(https://drive.google.com/drive/folders/1tMW7kRVrirU9R4wxlCq58YDRba-UZ3Tr?usp=sharing). Please ask for access