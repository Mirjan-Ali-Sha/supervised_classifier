# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SupervisedClassification
                                 A QGIS plugin
 A plugin to classify selected raster file with reference
                              -------------------
        begin                : 2024-06-15
        git sha              : $Format:%H$
        email                : mastools.help@gmail.com
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QToolBar
from .classification_dialog import ClassificationDialog
from osgeo import ogr, gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import numpy as np
from qgis.core import QgsRasterLayer, QgsProject
import os
import subprocess
import sys
from .resources_rc import *
import time
import pandas as pd

class SupervisedClassification:
    """QGIS Plugin Implementation."""
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr(u'&MAS Raster Processing')  # Common menu name
        self.toolbar = None
        self.first_start = None
        self.install_required_packages()


    def tr(self, message):
        return QCoreApplication.translate('SupervisedClassification', message)

    def add_action(self,icon_path,text,callback, enabled_flag=True, add_to_menu=True, add_to_toolbar=True, status_tip=None, whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        # if status_tip is not None:
        #     action.setStatusTip(status_tip)

        # if whats_this is not None:
        #     action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)
        if add_to_menu:
            self.iface.addPluginToRasterMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = ':/supervised.png'
        # Check if the toolbar already exists, if not create it
        self.toolbar = self.iface.mainWindow().findChild(QToolBar, 'MASRasterProcessingToolbar')
        if self.toolbar is None:
            self.toolbar = self.iface.addToolBar(u'MAS Raster Processing')
            self.toolbar.setObjectName('MASRasterProcessingToolbar')

        self.action_SpvClassification = QAction(QIcon(icon_path), u"&Supervised Classifier", self.iface.mainWindow())
        self.action_SpvClassification.triggered.connect(self.run)
        self.iface.addPluginToRasterMenu(self.menu, self.action_SpvClassification)
        self.toolbar.addAction(self.action_SpvClassification)
        self.actions.append(self.action_SpvClassification)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&MAS Raster Processing'), action)
            self.iface.removeToolBarIcon(action)
        if self.toolbar:
            del self.toolbar

    def install_required_packages(self):
        try:
            import numpy
            import scipy
            import sklearn
        except ImportError:
            requirements_path = os.path.join(self.plugin_dir, 'requirements.txt')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

    def run(self):
        self.show_classification_dialog()

    def show_classification_dialog(self):
        self.dialog = ClassificationDialog()
        self.dialog.classify_signal.connect(self.classify)
        self.dialog.on_method_change()
        self.dialog.exec_()

    def update_progress(self, step, total_steps):
        progress = int((step / total_steps) * 100)
        self.dialog.progress_bar.setValue(progress)
        QCoreApplication.processEvents()  # Update the GUI

    def classify(self, image_path, shapefile_path, output_folder, method, open_in_qgis, label_field, save_model, num_iterations):
        if not image_path or not output_folder:
            QMessageBox.critical(None, "Missing Input", "Please provide both input image and output folder.")
            return

        start_time = time.time()  # Start timer

        try:
            X_train, y_train, label_encoder = self.prepare_training_data(shapefile_path, image_path, label_field)
            if X_train.size == 0 or y_train.size == 0:
                raise Exception("No valid training data found.")
            classifier = self.get_classifier(method, num_iterations)
            X_test = self.prepare_test_data(image_path)
            if X_test.size == 0:
                raise Exception("No valid test data found.")
            if method == "Minimum Distance":
                predictions = self.classify_minimum_distance((X_train, y_train), X_test)
            else:
                for i in range(num_iterations):
                    classifier.fit(X_train, y_train)
                    self.update_progress(i + 1, num_iterations)
                predictions = classifier.predict(X_test)
            predictions = label_encoder.inverse_transform(predictions)
            classified_image_path, label_mappings = self.save_classified_image(predictions, output_folder, image_path, label_encoder, method)
            if save_model:
                accuracy, class_distribution = self.save_model_info(classifier, output_folder, method, X_train, y_train, label_encoder)

            if open_in_qgis:
                self.open_output_in_qgis(classified_image_path)

            end_time = time.time()  # End timer
            elapsed_time = end_time - start_time

            info = f"Classification process completed in {elapsed_time:.2f} seconds.\nLabel mappings: {label_mappings}\n"
            if save_model:
                info += f"Trained model accuracy: {accuracy}\n"
                info += f"Class distribution in training data: {class_distribution}\n"
            
            self.dialog.show_processing_info(info)
            QMessageBox.information(None, "Classification Successful", f"The image was classified successfully in {elapsed_time:.2f} seconds.")
        except Exception as e:
            QMessageBox.critical(None, "Classification Failed", f"Failed to classify the image: {str(e)}")

    def get_classifier(self, method, num_iterations):
        if method == "Random Forest":
            return RandomForestClassifier(n_estimators=num_iterations)
        elif method == "SVM":
            return SVC(max_iter=num_iterations)
        elif method == "KNN (Not Good)":
            return KNeighborsClassifier()
        elif method == "Minimum Distance":
            return None  # No classifier object needed
        else:
            raise Exception("Invalid method selected")

    def classify_minimum_distance(self, train_data, X_test):
        X_train, y_train = train_data
        class_means = np.array([X_train[y_train == k].mean(axis=0) for k in np.unique(y_train)])
        return np.argmin(cdist(X_test, class_means), axis=1)

    def prepare_training_data(self, shapefile_path, image_path, label_field):
        shapefile_ds = ogr.Open(shapefile_path)
        if not shapefile_ds:
            raise Exception(f"Failed to open shapefile: {shapefile_path}")

        image_ds = gdal.Open(image_path)
        if not image_ds:
            raise Exception(f"Failed to open image: {image_path}")

        layer = shapefile_ds.GetLayer()
        if not layer:
            raise Exception("Failed to get layer from shapefile")

        geotransform = image_ds.GetGeoTransform()
        x_min = 0
        x_max = image_ds.RasterXSize
        y_min = 0
        y_max = image_ds.RasterYSize

        bands = [image_ds.GetRasterBand(i+1) for i in range(image_ds.RasterCount)]
        if not bands:
            raise Exception("Failed to get raster bands from image")

        X_train = []
        y_train = []

        label_encoder = LabelEncoder()
        missing_classes = set()

        for feature in layer:
            geom = feature.GetGeometryRef()
            if not geom:
                continue
            label = feature.GetField(label_field)
            if label is None:
                continue
            x, y = geom.Centroid().GetX(), geom.Centroid().GetY()
            pixel_x = int((x - geotransform[0]) / geotransform[1])
            pixel_y = int((y - geotransform[3]) / geotransform[5])
            if not (x_min <= pixel_x < x_max and y_min <= pixel_y < y_max):
                print(f"Skipping point ({x}, {y}) which maps to pixel ({pixel_x}, {pixel_y})")
                continue
            pixel_values = []
            for band in bands:
                band_array = band.ReadAsArray(pixel_x, pixel_y, 1, 1)
                if band_array is None:
                    print(f"Failed to read band array at ({pixel_x}, {pixel_y})")
                    continue
                pixel_values.append(band_array[0][0])
            if len(pixel_values) == len(bands):
                X_train.append(pixel_values)
                y_train.append(label)
            else:
                missing_classes.add(label)

        if missing_classes:
            warning_message = f"Warning: The following classes have no training samples: {missing_classes}. Take more samples for {missing_classes} class."
            QMessageBox.warning(None, "Missing Classes", warning_message)
            print(warning_message)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if X_train.size == 0 or y_train.size == 0:
            return X_train, y_train, label_encoder

        y_train = label_encoder.fit_transform(y_train)

        print("Training data prepared:")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("Labels:", list(label_encoder.classes_))

        return X_train, y_train, label_encoder

    def prepare_test_data(self, image_path):
        image_ds = gdal.Open(image_path)
        if not image_ds:
            raise Exception(f"Failed to open image: {image_path}")

        bands = [image_ds.GetRasterBand(i+1) for i in range(image_ds.RasterCount)]
        if not bands:
            raise Exception("Failed to get raster bands from image")

        X_test = []
        for row in range(image_ds.RasterYSize):
            for col in range(image_ds.RasterXSize):
                pixel_values = []
                for band in bands:
                    band_array = band.ReadAsArray(col, row, 1, 1)
                    if band_array is None:
                        raise Exception(f"Failed to read band array at ({col}, {row})")
                    pixel_values.append(band_array[0][0])
                X_test.append(pixel_values)

        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        print("Test data prepared:")
        print("X_test shape:", X_test.shape)

        return X_test

    def save_classified_image(self, predictions, output_folder, image_path, label_encoder, method):
        image_ds = gdal.Open(image_path)
        if not image_ds:
            raise Exception(f"Failed to open image: {image_path}")

        driver = gdal.GetDriverByName("GTiff")
        suffix = method.replace(" ", "_").lower()
        output_path = os.path.join(output_folder, f"classified_image_{suffix}.tif")
        output_ds = driver.Create(output_path, image_ds.RasterXSize, image_ds.RasterYSize, 1, gdal.GDT_Int32)
        output_ds.SetGeoTransform(image_ds.GetGeoTransform())
        output_ds.SetProjection(image_ds.GetProjection())
        output_band = output_ds.GetRasterBand(1)
        label_to_int = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}  # Start numbering from 1
        classified_image = np.vectorize(label_to_int.get)(predictions).reshape(image_ds.RasterYSize, image_ds.RasterXSize)
        output_band.WriteArray(classified_image)
        output_band.FlushCache()
        output_ds = None
        label_mappings = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}

        print("Classified image saved to:", output_path)

        return output_path, label_mappings

    def save_model_info(self, classifier, output_folder, method, X_train, y_train, label_encoder):
        import joblib
        suffix = method.replace(" ", "_").lower()
        model_path = os.path.join(output_folder, f"trained_model_{suffix}.pkl")
        if method != "Minimum Distance":
            joblib.dump(classifier, model_path)

        from sklearn.metrics import accuracy_score, classification_report
        y_pred = classifier.predict(X_train) if method != "Minimum Distance" else self.classify_minimum_distance((X_train, y_train), X_train)
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, target_names=label_encoder.classes_)

        model_info_path = os.path.join(output_folder, f"model_info_{suffix}.txt")
        class_distribution = pd.Series(y_train).value_counts().to_string()
        label_mappings = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}

        with open(model_info_path, 'w') as file:
            if method != "Minimum Distance":
                file.write(f"Model Path: {model_path}\n")
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"Classification Report:\n{report}\n")
            file.write(f"Labels: {list(label_encoder.classes_)}\n")
            file.write(f"Class distribution in training data:\n{class_distribution}\n")
            file.write(f"Label mappings (label to numerical value): {label_mappings}\n")

        if method not in ["Minimum Distance", "Maximum Likelihood (Not Working)"]:
            model_info_file = os.path.join(os.path.dirname(__file__), 'model_paths.txt')
            with open(model_info_file, 'a') as file:
                file.write(model_info_path + '\n')

        print("Model info saved to:", model_info_path)

        return accuracy, class_distribution

    def load_model_info(self, file_path):
        model_info = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.split(": ", 1)
                model_info[key.strip()] = value.strip()
        return model_info

    def load_model(self, model_path, method):
        if method == "Minimum Distance":
            return None  # No model to load for Minimum Distance method
        import joblib
        return joblib.load(model_path)

    def open_output_in_qgis(self, classified_image_path):
        layer = QgsRasterLayer(classified_image_path, "Classified Image")
        if not layer.isValid():
            raise Exception("Failed to load classified image in QGIS.")
        QgsProject.instance().addMapLayer(layer)

    def show_processing_info(self, info):
        self.dialog.processing_info_label.setText(info)
