# Supervised Classifier
A plugin to classify selected raster file with reference
The Supervised Classifier Plugin for QGIS is a powerful tool designed to facilitate the classification of satellite images using unsupervised learning algorithms. 
This plugin provides an easy-to-use interface for loading satellite images and selecting from a variety of supervised classification methods, including "Minimum Distance", "Random Forest", "SVM" and KNN (right now KNN algorithm not working properly, I'm working on it). Also working on "Maximum Likelihood" algorithm as well.
## Key Features: 
1. __User-Friendly Interface:__ Intuitive dialog for selecting classification parameters. 
2. __Multiple Algorithms:__ Support for _"Minimum Distance"_ (most effective), _"Random Forest"_, _"SVM"_ and _"KNN"_ (not good) algorithms. 
3. __Seamless Integration:__ Directly integrates with QGIS, allowing for easy access and visualization of classification results. 
4. __Flexible:__ Handles different types of satellite images and provides robust classification results. 
This plugin is ideal for remote sensing professionals, GIS analysts, and researchers looking to perform efficient and accurate unsupervised classification of satellite imagery within the QGIS environment. 
### To use this Tool follow the below steps: 
1. Click on the tool or chose "__Raster__" menu --> "__MAS Raster Processing__" menu item --> "__Supervised Classifier__" option. 
2. Select 'Stack Image' or Image as Input and select output folder name. 
3. Select classification method. 
4. Adjust all parameters according to your needs. Select the field of the _Reference Shapefile_ which has the labels for the classification and check mark on __"Want to save the trained model?"__ if you wants save the trained model (except "Minimum Distance" model). Select the _"No of Iterations:"_ to trained the model (except __"Minimum Distance"__ model).
5. Decide do you wants to open the output or not (By marking on _"Do you want to open output in QGIS Interface?"_). 
6. Click on __"Classify"__ button. 
###
** **Note:** After installation make sure the following points; 
1. Check Mark the Installed plugins (under 'Manage and Install Plugins...' menu) 
2. Check Mark __'MAS Raster Processing'__ toolbar (by right click on toolbar).
