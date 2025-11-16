# Supervised Classifier:
A plugin to classify selected raster file with reference
The Supervised Classifier Plugin for QGIS is a powerful tool designed to facilitate the classification of satellite images using unsupervised learning algorithms. 
This plugin provides an easy-to-use interface for loading satellite images and selecting from a variety of supervised classification methods, including "Minimum Distance", "Random Forest", "SVM" and KNN (right now KNN algorithm not working properly, I'm working on it). Also working on "Maximum Likelihood" algorithm as well.
## Key Features:
- User-Friendly Interface with Batch Processing
- Multiple Classification Algorithms (Random Forest, SVM, KNN, Minimum Distance)
- Pre-trained Model Support - Classify multiple rasters using saved models
- Band Selection - Choose specific bands for classification
- Multiple Label Field Support - Handle complex reference shapefiles
- Model Saving & Export - Save trained models with detailed info files
- Label Mapping Export - Export classification labels as CSV/JSON
- Progress Tracking & Error Handling
- Seamless QGIS Integration

This plugin is ideal for remote sensing professionals, GIS analysts, and researchers looking to perform efficient and accurate unsupervised classification of satellite imagery within the QGIS environment. 
### To use this Tool follow the below steps: 
1. Click on the tool or chose "__Raster__" menu --> "__MAS Raster Processing__" menu item --> "__Supervised Classifier__" option. 
2. Select 'Stack Image' or Image as Input and select output folder name. 
3. Select classification method. 
4. Adjust all parameters according to your needs. Select the field of the _Reference Shapefile_ which has the labels for the classification and check mark on __"Want to save the trained model?"__ if you wants save the trained model (except "Minimum Distance" model). Select the _"No of Iterations:"_ to trained the model (except __"Minimum Distance"__ model).
5. Decide do you wants to open the output or not (By marking on _"Do you want to open output in QGIS Interface?"_). 
6. Click on __"Classify"__ button. 
### Follow the below steps for Version >= 1.0.0:

**Training New Model Mode:**
1. Open the tool: __Raster__ menu → "__MAS Raster Processing__" → "__Supervised Classifier__"
2. Keep "__Do you want to use a pre-trained model?__" unchecked
3. Select input rasters from the table or add new ones using "__Add Raster from File__"
4. Select reference shapefiles and choose label fields using the "__Select Label Fields__" button
5. Choose __output folder__ (or check "__Save output in same folder as input?__")
6. Select __classification method__ (Random Forest, SVM, KNN, or Minimum Distance)
7. Check "__Wants to save trained model?__" to save the model for future use
8. Set "__No of Iterations__" for training (except Minimum Distance)
9. Optionally check "__Do you want to open output in QGIS Interface?__"
10. Click the "__Classify__" button
11. View label mappings dialog and export if needed
---
## 📹 Video Tutorials

<div align="center">

### Training New Model Mode
<img src="https://github.com/Mirjan-Ali-Sha/supervised_classifier/blob/main/assets/Supervised_Classifier.gif" width="750" alt="Installation Tutorial"/>

*Learn how to train the New Model Mode*
### Data Labels:
You can use one Label column (label) with different numbers (1,2,3,4) or with different strings (Water, Forest, Settlement, FarmLand) or Multiple label columns (in this video, each column has 0 and 1 as values to identify its class). For more details, see the video below <i>(which also shows how to use a pre-trained model, no matter which classifier it has to be a .pkl file and supported by `scikit-learn`).</i>

---

### Pre-trained Model Mode:
<img src="https://github.com/Mirjan-Ali-Sha/supervised_classifier/blob/main/assets/Supervised_Classifier_pt_Model.gif" width="750" alt="Download Tutorial"/>

*Discover how to use Pre-trained Model Mode*

</div>

**Pre-trained Model Mode:**
1. Open the tool and check "__Do you want to use a pre-trained model?__"
2. Select input rasters from the table
3. Add pre-trained model files (.pkl) using the "__Add Model from File__" button
4. If multiple models are selected, confirm the warning about the total output files
5. Choose output folder
6. Optionally check "__Do you want to open output in QGIS Interface?__"
7. Click the "__Classify__" button
8. Check model info files for label mapping details

###
** **Note:** After installation make sure the following points; 
1. Check Mark the Installed plugins (under 'Manage and Install Plugins...' menu) 
2. Check Mark __'MAS Raster Processing'__ toolbar (by right click on toolbar).
#### Fixed the Issues (if get errors like below): 
[ModuleNotFoundError: No module named 'sklearn' #1](https://github.com/Mirjan-Ali-Sha/supervised_classifier/issues/1)

#### **Solution:** @ModuleNotFoundError: No module named 'sklearn'

1. For windows OS follow the below steps:

- Search and open "OSGeo4W Shell".
- Run the below command:
- `pip install scikit-learn`
- Restart the QGIS and mark on "Supervised Classifier" (under 'Installed Plugin') from "Manage and Install Plugins..."

2. For Other OS follow the below steps:

- Open Terminal and Run the below command:
- `pip install scikit-learn`
