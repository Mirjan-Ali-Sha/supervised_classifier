from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, 
    QFileDialog, QMessageBox, QSpinBox, QProgressBar
)
from PyQt5.QtCore import pyqtSignal, Qt
import os
import numpy as np
from osgeo import ogr, gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import time
import pandas as pd

class ClassificationDialog(QDialog):
    classify_signal = pyqtSignal(str, str, str, str, bool, str, bool, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Supervised Classification")
        self.resize(500, 600)
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Input Image:"))
        self.image_path = QLineEdit()
        layout.addWidget(self.image_path)
        image_browse_button = QPushButton("Browse")
        image_browse_button.clicked.connect(lambda: self.browse_file(self.image_path, "Select Image", "Images (*.tif *.jpg *.png)"))
        layout.addWidget(image_browse_button)
        
        layout.addWidget(QLabel("Reference Shapefile:"))
        self.shapefile_path = QLineEdit()
        layout.addWidget(self.shapefile_path)
        shapefile_browse_button = QPushButton("Browse")
        shapefile_browse_button.clicked.connect(self.browse_shapefile)
        layout.addWidget(shapefile_browse_button)
        
        layout.addWidget(QLabel("Output Folder:"))
        self.output_folder = QLineEdit()
        layout.addWidget(self.output_folder)
        output_browse_button = QPushButton("Browse")
        output_browse_button.clicked.connect(self.browse_folder)
        layout.addWidget(output_browse_button)
        
        layout.addWidget(QLabel("Select Label Field:"))
        self.label_field_combo = QComboBox()
        layout.addWidget(self.label_field_combo)
        
        layout.addWidget(QLabel("Classification Method:"))
        self.classification_methods = QComboBox()
        self.classification_methods.addItems(["Minimum Distance", "Random Forest", "SVM", "KNN (Not Good)", "Maximum Likelihood (Not Working)"])
        self.classification_methods.currentTextChanged.connect(self.on_method_change)
        layout.addWidget(self.classification_methods)
        
        self.save_model_checkbox = QCheckBox("Wants to save trained model?")
        self.save_model_checkbox.setChecked(True)
        self.save_model_checkbox.clicked.connect(self.on_save_model_checkbox_clicked)
        layout.addWidget(self.save_model_checkbox)
        
        layout.addWidget(QLabel("No of Iterations:"))
        self.num_iterations_spinbox = QSpinBox()
        self.num_iterations_spinbox.setMinimum(1)
        self.num_iterations_spinbox.setMaximum(999)
        self.num_iterations_spinbox.setValue(100)
        layout.addWidget(self.num_iterations_spinbox)
        
        self.open_in_qgis = QCheckBox("Do you want to open output in QGIS Interface?")
        layout.addWidget(self.open_in_qgis)
        
        classify_button = QPushButton("Classify")
        classify_button.clicked.connect(self.classify)
        layout.addWidget(classify_button)

        self.processing_info_label = QLabel()
        self.processing_info_label.setWordWrap(True)
        layout.addWidget(self.processing_info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        self.resize_ui()

    def browse_file(self, line_edit, title, filter):
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if file_path:
            line_edit.setText(file_path)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder.setText(folder_path)

    def browse_shapefile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Shapefile", "", "Shapefiles (*.shp)")
        if file_path:
            self.shapefile_path.setText(file_path)
            self.update_label_fields(file_path)

    def update_label_fields(self, shapefile_path):
        shapefile_ds = ogr.Open(shapefile_path)
        if shapefile_ds:
            layer = shapefile_ds.GetLayer()
            if layer:
                self.label_field_combo.clear()
                layer_def = layer.GetLayerDefn()
                for i in range(layer_def.GetFieldCount()):
                    field_def = layer_def.GetFieldDefn(i)
                    self.label_field_combo.addItem(field_def.GetName())

    def on_method_change(self):
        method = self.classification_methods.currentText()
        if method in ["Minimum Distance", "Maximum Likelihood (Not Working)"]:
            self.save_model_checkbox.setChecked(False)
            self.save_model_checkbox.setEnabled(False)
            self.num_iterations_spinbox.setVisible(False)
        else:
            self.save_model_checkbox.setChecked(True)
            self.save_model_checkbox.setEnabled(True)
            self.num_iterations_spinbox.setVisible(True)
        self.resize_ui()

    def on_save_model_checkbox_clicked(self):
        if not self.save_model_checkbox.isEnabled():
            method = self.classification_methods.currentText()
            QMessageBox.warning(self, "Unsupported Option", f"Your selected model {method} does not support this option.")
            self.save_model_checkbox.setChecked(False)

    def classify(self):
        image_path = self.image_path.text()
        shapefile_path = self.shapefile_path.text()
        output_folder = self.output_folder.text()
        method = self.classification_methods.currentText()
        open_in_qgis = self.open_in_qgis.isChecked()
        label_field = self.label_field_combo.currentText()
        save_model = self.save_model_checkbox.isChecked()
        num_iterations = self.num_iterations_spinbox.value()

        if not image_path or not output_folder:
            QMessageBox.critical(self, "Missing Input", "Please provide both input image and output folder.")
            return

        self.progress_bar.setValue(0)
        self.processing_info_label.setText("Classification process started...")
        start_time = time.time()
        self.classify_signal.emit(image_path, shapefile_path, output_folder, method, open_in_qgis, label_field, save_model, num_iterations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.progress_bar.setValue(100)

    def resize_ui(self):
        if self.num_iterations_spinbox.isVisible():
            self.resize(400, 600)
        else:
            self.resize(400, 560)

    def show_processing_info(self, info):
        self.processing_info_label.setText(info)
