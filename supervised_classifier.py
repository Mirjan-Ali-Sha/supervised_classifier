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
from qgis.PyQt.QtWidgets import (QAction, QMessageBox, QToolBar, QDialog, QVBoxLayout, 
                                 QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
                                 QFileDialog, QHeaderView, QLabel)
from qgis.core import QgsRasterLayer, QgsProject
import os
import sys
from .resources_rc import *
import time
import json

# Flag to track if dependencies are available
DEPENDENCIES_AVAILABLE = False

# Try to import required packages
try:
    from .classification_dialog import ClassificationDialog
    from osgeo import ogr, gdal
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from scipy.spatial.distance import cdist
    import numpy as np
    import joblib
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Dependencies not available - will be handled in __init__
    pass


class LabelMappingDialog(QDialog):
    """Dialog to display and export label mappings"""
    def __init__(self, label_mappings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Mappings")
        self.resize(500, 400)
        self.label_mappings = label_mappings
        
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel("Output Raster Values and Label Names:")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title_label)
        
        # Table to display mappings
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(2)
        self.mapping_table.setHorizontalHeaderLabels(["Raster Value", "Label Name"])
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mapping_table.setRowCount(len(label_mappings))
        
        # Populate table
        sorted_mappings = sorted(label_mappings.items(), key=lambda x: x[1])
        for row, (label_name, raster_value) in enumerate(sorted_mappings):
            # Raster value
            value_item = QTableWidgetItem(str(raster_value))
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 0, value_item)
            
            # Label name
            name_item = QTableWidgetItem(str(label_name))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 1, name_item)
        
        layout.addWidget(self.mapping_table)
        
        # Export buttons
        button_layout = QHBoxLayout()
        
        export_csv_button = QPushButton("Export as CSV")
        export_csv_button.clicked.connect(self.export_as_csv)
        button_layout.addWidget(export_csv_button)
        
        export_json_button = QPushButton("Export as JSON")
        export_json_button.clicked.connect(self.export_as_json)
        button_layout.addWidget(export_json_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def export_as_csv(self):
        """Export label mappings as CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Label Mappings as CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                df = pd.DataFrame(list(self.label_mappings.items()), 
                                 columns=['Label_Name', 'Raster_Value'])
                df = df.sort_values('Raster_Value')
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful", 
                                       f"Label mappings exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                                    f"Failed to export CSV: {str(e)}")
    
    def export_as_json(self):
        """Export label mappings as JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Label Mappings as JSON", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.label_mappings, f, indent=4)
                QMessageBox.information(self, "Export Successful", 
                                       f"Label mappings exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                                    f"Failed to export JSON: {str(e)}")


class SupervisedClassification:
    """QGIS Plugin Implementation."""
    
    # Required packages: {pip_package_name: import_module_name}
    REQUIRED_PACKAGES = {
        'scikit-learn': 'sklearn',
        'scipy': 'scipy',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'joblib': 'joblib'
    }
    
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr(u'&MAS Raster Processing')
        self.toolbar = None
        self.first_start = None
        self.dependencies_ok = DEPENDENCIES_AVAILABLE
        # Dependencies are checked when user clicks the plugin, not at startup

    def _check_and_install_dependencies(self):
        """Check and install missing dependencies using the robust installer"""
        try:
            from .dependency_installer import DependencyInstaller
            
            installer = DependencyInstaller(self.iface, self.REQUIRED_PACKAGES)
            installer.PLUGIN_NAME = "Supervised Classifier"
            
            if installer.check_and_install(silent_if_ok=True):
                # Dependencies installed successfully - try to reload them
                if self._reload_dependencies():
                    self.dependencies_ok = True
                    QMessageBox.information(
                        self.iface.mainWindow(),
                        "Ready to Use",
                        "Dependencies installed successfully!\n\n"
                        "The plugin is now ready to use."
                    )
                else:
                    # Reload failed - need restart
                    self.dependencies_ok = False
            else:
                self.dependencies_ok = False
        except Exception as e:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Failed to check dependencies: {str(e)}",
                "Supervised Classifier",
                Qgis.Warning
            )
            self.dependencies_ok = False
    
    def _reload_dependencies(self):
        """
        Try to reload/import all required dependencies after installation.
        Returns True if all imports succeed, False otherwise.
        """
        global DEPENDENCIES_AVAILABLE
        global ClassificationDialog, ogr, gdal
        global RandomForestClassifier, SVC, KNeighborsClassifier, LabelEncoder
        global cdist, np, joblib, pd
        
        try:
            # Import all required modules
            from .classification_dialog import ClassificationDialog as CD
            from osgeo import ogr as _ogr, gdal as _gdal
            from sklearn.ensemble import RandomForestClassifier as RFC
            from sklearn.svm import SVC as _SVC
            from sklearn.neighbors import KNeighborsClassifier as KNC
            from sklearn.preprocessing import LabelEncoder as LE
            from scipy.spatial.distance import cdist as _cdist
            import numpy as _np
            import joblib as _joblib
            import pandas as _pd
            
            # Update global references
            ClassificationDialog = CD
            ogr = _ogr
            gdal = _gdal
            RandomForestClassifier = RFC
            SVC = _SVC
            KNeighborsClassifier = KNC
            LabelEncoder = LE
            cdist = _cdist
            np = _np
            joblib = _joblib
            pd = _pd
            
            DEPENDENCIES_AVAILABLE = True
            
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                "Successfully reloaded all dependencies",
                "Supervised Classifier",
                Qgis.Success
            )
            return True
            
        except ImportError as e:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage(
                f"Failed to reload dependencies: {str(e)}",
                "Supervised Classifier",
                Qgis.Warning
            )
            return False

    def tr(self, message):
        return QCoreApplication.translate('SupervisedClassification', message)

    def add_action(self, icon_path, text, callback, enabled_flag=True, add_to_menu=True, add_to_toolbar=True, status_tip=None, whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if add_to_toolbar:
            self.toolbar.addAction(action)
        if add_to_menu:
            self.iface.addPluginToRasterMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = ':/supervised.png'
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

    def run(self):
        # Check if dependencies are available
        if not self.dependencies_ok:
            # Try to reload dependencies first (they might have been installed)
            if self._reload_dependencies():
                self.dependencies_ok = True
            else:
                # Still not available - prompt for installation
                self._check_and_install_dependencies()
                if not self.dependencies_ok:
                    return
        
        self.show_classification_dialog()

    def show_classification_dialog(self):
        self.dialog = ClassificationDialog()
        self.dialog.classify_signal.connect(self.classify)
        self.dialog.on_method_change()
        self.dialog.exec_()

    def update_progress(self, step, total_steps):
        progress = int((step / total_steps) * 100)
        self.dialog.progress_bar.setValue(progress)
        QCoreApplication.processEvents()

    def classify(self, selected_rasters, selected_references, output_folder, method, open_in_qgis, save_model, num_iterations, use_pretrained_model):
        """
        Handle batch classification for multiple rasters and references
        """
        if not selected_rasters:
            QMessageBox.critical(None, "Missing Input", "Please select at least one raster image.")
            return
        
        if not selected_references:
            QMessageBox.critical(None, "Missing Input", "Please select at least one reference file.")
            return
        
        # Warning for multiple pre-trained models
        if use_pretrained_model and len(selected_references) > 1:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Multiple Models Selected")
            msg_box.setText(f"You have selected {len(selected_references)} pre-trained models.")
            msg_box.setInformativeText(
                "Each raster will be classified using ALL selected models.\n\n"
                f"Total outputs: {len(selected_rasters)} rasters × {len(selected_references)} models = "
                f"{len(selected_rasters) * len(selected_references)} files.\n\n"
                "Do you want to continue?"
            )
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            if msg_box.exec_() == QMessageBox.No:
                return
        
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        last_label_mappings = None
        
        try:
            if use_pretrained_model:
                # ========== PRE-TRAINED MODEL MODE ==========
                # Process: Model 1 → All Rasters → Model 2 → All Rasters
                total_items = len(selected_references) * len(selected_rasters)
                current_item = 0
                
                for model_idx, ref_info in enumerate(selected_references):
                    model_path = ref_info['path']
                    model_name = os.path.basename(model_path)
                    
                    print(f"\n{'='*70}")
                    print(f"LOADING MODEL {model_idx + 1}/{len(selected_references)}: {model_name}")
                    print(f"{'='*70}")
                    
                    # Load model once
                    classifier = self.load_model(model_path)
                    if not classifier:
                        print(f"❌ Failed to load model: {model_name}")
                        failed_count += len(selected_rasters)
                        current_item += len(selected_rasters)
                        continue
                    
                    # Get class names once
                    class_names = self.get_class_names_from_model(model_path)
                    if not class_names:
                        print(f"❌ Failed to get class names for: {model_name}")
                        failed_count += len(selected_rasters)
                        current_item += len(selected_rasters)
                        continue
                    
                    print(f"✓ Model loaded successfully")
                    print(f"✓ Classes: {class_names}")
                    
                    # Classify all rasters with this model
                    for raster_idx, raster_info in enumerate(selected_rasters):
                        current_item += 1
                        image_path = raster_info['path']
                        output_name = raster_info['output_name']
                        selected_bands = raster_info['selected_bands']
                        
                        # Modify output name to include model name
                        base_name, ext = os.path.splitext(output_name)
                        model_suffix = os.path.splitext(model_name)[0].replace('trained_model_', '')
                        modified_output_name = f"{base_name}_{model_suffix}{ext}"
                        
                        self.dialog.show_processing_info(
                            f"Model {model_idx + 1}/{len(selected_references)} - "
                            f"Raster {raster_idx + 1}/{len(selected_rasters)}: {raster_info['name']}"
                        )
                        QCoreApplication.processEvents()
                        
                        try:
                            # Determine output folder
                            if output_folder:
                                current_output_folder = output_folder
                            else:
                                current_output_folder = os.path.dirname(image_path)
                            
                            # Prepare test data
                            X_test = self.prepare_test_data(image_path, selected_bands)
                            if X_test.size == 0:
                                raise Exception("No test data")
                            
                            # Predict
                            predictions = classifier.predict(X_test)
                            
                            # Convert to label names
                            label_predictions = np.array([class_names[int(p)] for p in predictions])
                            
                            # Create temp encoder
                            class TempEncoder:
                                def __init__(self, classes):
                                    self.classes_ = classes
                            
                            encoder = TempEncoder(class_names)
                            
                            # Save
                            classified_image_path, label_mappings = self.save_classified_image(
                                label_predictions, current_output_folder, image_path, 
                                encoder, "Pretrained", modified_output_name
                            )
                            
                            last_label_mappings = label_mappings
                            
                            if open_in_qgis:
                                self.open_output_in_qgis(classified_image_path)
                            
                            processed_count += 1
                            print(f"✓ Saved: {os.path.basename(classified_image_path)}")
                        
                        except Exception as e:
                            failed_count += 1
                            error_msg = f"Failed {raster_info['name']}: {str(e)}"
                            print(f"❌ {error_msg}")
                            import traceback
                            traceback.print_exc()
                            self.dialog.show_processing_info(error_msg)
                            QCoreApplication.processEvents()
                        
                        # Update progress
                        progress = int((current_item / total_items) * 100)
                        self.dialog.progress_bar.setValue(progress)
                        QCoreApplication.processEvents()
            
            else:
                # ========== TRAINING MODE ==========
                # Process: Raster 1 → All References → Raster 2 → All References
                total_items = len(selected_rasters)
                
                for idx, raster_info in enumerate(selected_rasters):
                    image_path = raster_info['path']
                    output_name = raster_info['output_name']
                    selected_bands = raster_info['selected_bands']
                    
                    self.dialog.show_processing_info(f"Processing {idx + 1}/{total_items}: {raster_info['name']}")
                    QCoreApplication.processEvents()
                    
                    try:
                        # Determine output folder
                        if output_folder:
                            current_output_folder = output_folder
                        else:
                            current_output_folder = os.path.dirname(image_path)
                        
                        # Process each reference file
                        for ref_info in selected_references:
                            shapefile_path = ref_info['path']
                            label_fields = ref_info['selected_fields']
                            
                            if not label_fields:
                                raise Exception(f"No label fields selected for {ref_info['name']}")
                            
                            X_train, y_train, label_encoder = self.prepare_training_data(
                                shapefile_path, image_path, label_fields, selected_bands
                            )
                            
                            if X_train.size == 0 or y_train.size == 0:
                                raise Exception("No valid training data found.")
                            
                            classifier = self.get_classifier(method, num_iterations)
                            X_test = self.prepare_test_data(image_path, selected_bands)
                            
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
                            
                            classified_image_path, label_mappings = self.save_classified_image(
                                predictions, current_output_folder, image_path, label_encoder, method, output_name
                            )
                            
                            last_label_mappings = label_mappings
                            
                            if save_model:
                                self.save_model_info(
                                    classifier, current_output_folder, method, X_train, y_train, label_encoder
                                )
                            
                            if open_in_qgis:
                                self.open_output_in_qgis(classified_image_path)
                        
                        processed_count += 1
                    
                    except Exception as e:
                        failed_count += 1
                        error_msg = f"Failed to process {raster_info['name']}: {str(e)}"
                        print(f"❌ {error_msg}")
                        import traceback
                        traceback.print_exc()
                        self.dialog.show_processing_info(error_msg)
                        QCoreApplication.processEvents()
                    
                    # Update progress
                    progress = int((idx + 1) / total_items * 100)
                    self.dialog.progress_bar.setValue(progress)
                    QCoreApplication.processEvents()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if use_pretrained_model:
                total_expected = len(selected_rasters) * len(selected_references)
                info = f"Classification process completed in {elapsed_time:.2f} seconds.\n"
                info += f"Successfully processed: {processed_count}/{total_expected} classifications.\n"
            else:
                info = f"Classification process completed in {elapsed_time:.2f} seconds.\n"
                info += f"Successfully processed: {processed_count}/{len(selected_rasters)} raster(s).\n"
            
            if failed_count > 0:
                info += f"Failed: {failed_count} item(s)."
            
            self.dialog.show_processing_info(info)
            
            # Show appropriate completion message
            if processed_count > 0:
                if use_pretrained_model:
                    QMessageBox.information(
                        None, "Classification Complete", 
                        f"Processed {processed_count} classifications in {elapsed_time:.2f} seconds.\n\n"
                        "Please check the model info files for label mapping details."
                    )
                elif last_label_mappings:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Classification Complete")
                    msg_box.setText(f"Processed {processed_count}/{len(selected_rasters)} raster(s) in {elapsed_time:.2f} seconds.")
                    msg_box.setInformativeText("Would you like to view the label mappings?")
                    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg_box.setDefaultButton(QMessageBox.Yes)
                    
                    if msg_box.exec_() == QMessageBox.Yes:
                        mapping_dialog = LabelMappingDialog(last_label_mappings)
                        mapping_dialog.exec_()
                else:
                    QMessageBox.information(None, "Classification Complete", 
                                        f"Processed {processed_count}/{len(selected_rasters)} raster(s) in {elapsed_time:.2f} seconds.")
        
        except Exception as e:
            QMessageBox.critical(None, "Classification Failed", f"Critical error: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_class_names_from_model(self, model_path):
        """Extract class names from model info file"""
        info_path = model_path.replace('.pkl', '_info.txt')
        if not os.path.exists(info_path):
            info_path = model_path.replace('trained_model_', 'model_info_').replace('.pkl', '.txt')
        
        if not os.path.exists(info_path):
            print(f"Model info not found: {info_path}")
            return None
        
        try:
            with open(info_path, 'r') as f:
                for line in f:
                    if line.startswith("Labels:"):
                        import ast
                        return ast.literal_eval(line.split("Labels:")[1].strip())
        except Exception as e:
            print(f"Error reading model info: {e}")
        
        return None

    def load_model(self, model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Model loading error: {e}")
            return None

    def get_classifier(self, method, num_iterations):
        if method == "Random Forest":
            return RandomForestClassifier(n_estimators=num_iterations)
        elif method == "SVM":
            return SVC(max_iter=num_iterations)
        elif method == "KNN (Not Good)":
            return KNeighborsClassifier()
        elif method == "Minimum Distance":
            return None
        else:
            raise Exception("Invalid method selected")

    def classify_minimum_distance(self, train_data, X_test):
        X_train, y_train = train_data
        class_means = np.array([X_train[y_train == k].mean(axis=0) for k in np.unique(y_train)])
        return np.argmin(cdist(X_test, class_means), axis=1)

    def prepare_training_data(self, shapefile_path, image_path, label_fields, selected_bands=None):
        """Prepare training data with support for multiple label columns"""
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
        x_min, x_max = 0, image_ds.RasterXSize
        y_min, y_max = 0, image_ds.RasterYSize

        # Get bands
        if selected_bands:
            bands = [image_ds.GetRasterBand(i) for i in selected_bands]
        else:
            bands = [image_ds.GetRasterBand(i+1) for i in range(image_ds.RasterCount)]

        if not bands:
            raise Exception("Failed to get raster bands")

        X_train, y_train = [], []
        label_encoder = LabelEncoder()
        
        # Handle multiple label fields
        if isinstance(label_fields, str):
            label_fields = [label_fields]
        
        multiple_labels = len(label_fields) > 1

        for feature in layer:
            geom = feature.GetGeometryRef()
            if not geom:
                continue
            
            # Get label
            if multiple_labels:
                label = None
                for field_name in label_fields:
                    if feature.GetField(field_name) == 1:
                        label = field_name
                        break
                if not label:
                    continue
            else:
                label = feature.GetField(label_fields[0])
                if label is None:
                    continue
            
            x, y = geom.Centroid().GetX(), geom.Centroid().GetY()
            pixel_x = int((x - geotransform[0]) / geotransform[1])
            pixel_y = int((y - geotransform[3]) / geotransform[5])
            
            if not (x_min <= pixel_x < x_max and y_min <= pixel_y < y_max):
                continue
            
            pixel_values = []
            valid = True
            for band in bands:
                arr = band.ReadAsArray(pixel_x, pixel_y, 1, 1)
                if arr is None:
                    valid = False
                    break
                pixel_values.append(arr[0, 0])
            
            if valid and len(pixel_values) == len(bands):
                X_train.append(pixel_values)
                y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if X_train.size == 0 or y_train.size == 0:
            return X_train, y_train, label_encoder

        y_train = label_encoder.fit_transform(y_train)
        
        print(f"Training data: {X_train.shape}, Labels: {list(label_encoder.classes_)}")

        return X_train, y_train, label_encoder

    def prepare_test_data(self, image_path, selected_bands=None):
        image_ds = gdal.Open(image_path)
        if not image_ds:
            raise Exception(f"Failed to open image: {image_path}")

        if selected_bands:
            bands = [image_ds.GetRasterBand(i) for i in selected_bands]
        else:
            bands = [image_ds.GetRasterBand(i+1) for i in range(image_ds.RasterCount)]

        if not bands:
            raise Exception("Failed to get raster bands")

        X_test = []
        for row in range(image_ds.RasterYSize):
            for col in range(image_ds.RasterXSize):
                pixel_values = []
                for band in bands:
                    arr = band.ReadAsArray(col, row, 1, 1)
                    if arr is None:
                        raise Exception(f"Failed to read band at ({col}, {row})")
                    pixel_values.append(arr[0, 0])
                X_test.append(pixel_values)

        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        return X_test

    def save_classified_image(self, predictions, output_folder, image_path, label_encoder, method, output_name=None):
        image_ds = gdal.Open(image_path)
        if not image_ds:
            raise Exception(f"Failed to open image: {image_path}")

        driver = gdal.GetDriverByName("GTiff")
        
        if output_name:
            output_path = os.path.join(output_folder, output_name)
        else:
            suffix = method.replace(" ", "_").lower()
            output_path = os.path.join(output_folder, f"classified_image_{suffix}.tif")
        
        output_ds = driver.Create(output_path, image_ds.RasterXSize, image_ds.RasterYSize, 1, gdal.GDT_Int32)
        output_ds.SetGeoTransform(image_ds.GetGeoTransform())
        output_ds.SetProjection(image_ds.GetProjection())
        output_band = output_ds.GetRasterBand(1)
        
        # Create label to int mapping
        if isinstance(label_encoder, dict):
            label_to_int = label_encoder
        else:
            label_to_int = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}
        
        # Convert predictions to raster values
        classified_image = np.vectorize(label_to_int.get)(predictions).reshape(
            image_ds.RasterYSize, image_ds.RasterXSize
        )
        output_band.WriteArray(classified_image)
        output_band.FlushCache()
        output_ds = None
        
        # Create label mappings
        if isinstance(label_encoder, dict):
            label_mappings = label_encoder
        else:
            label_mappings = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}

        return output_path, label_mappings

    def save_model_info(self, classifier, output_folder, method, X_train, y_train, label_encoder):
        suffix = method.replace(" ", "_").lower()
        model_path = os.path.join(output_folder, f"trained_model_{suffix}.pkl")
        joblib.dump(classifier, model_path)

        from sklearn.metrics import accuracy_score, classification_report
        
        if method == "Minimum Distance":
            y_pred = self.classify_minimum_distance((X_train, y_train), X_train)
        else:
            y_pred = classifier.predict(X_train)
        
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, target_names=label_encoder.classes_)

        model_info_path = os.path.join(output_folder, f"model_info_{suffix}.txt")
        class_distribution = pd.Series(y_train).value_counts().to_string()
        label_mappings = {label: i + 1 for i, label in enumerate(label_encoder.classes_)}

        with open(model_info_path, 'w') as file:
            file.write(f"Model Path: {model_path}\n")
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"Classification Report:\n{report}\n")
            file.write(f"Labels: {list(label_encoder.classes_)}\n")
            file.write(f"Class distribution in training data:\n{class_distribution}\n")
            file.write(f"Label mappings (label to numerical value): {label_mappings}\n")

        print(f"Model saved: {model_path}")
        return accuracy, class_distribution

    def open_output_in_qgis(self, classified_image_path):
        layer = QgsRasterLayer(classified_image_path, "Classified Image")
        if not layer.isValid():
            raise Exception("Failed to load classified image in QGIS.")
        QgsProject.instance().addMapLayer(layer)

    def show_processing_info(self, info):
        self.dialog.processing_info_label.setText(info)
