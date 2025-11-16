from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QSpinBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget
)
from PyQt5.QtCore import pyqtSignal, Qt
import os
from osgeo import ogr, gdal
from qgis.core import QgsProject, QgsMapLayerType


class BandSelectionDialog(QDialog):
    """Dialog for selecting bands from a raster"""
    def __init__(self, bands_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Bands")
        self.resize(400, 300)
        self.bands_info = bands_info
        self.selected_bands = []
        
        layout = QVBoxLayout()
        
        # Select/Unselect All button
        button_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Select All")
        self.toggle_button.clicked.connect(self.toggle_all)
        button_layout.addWidget(self.toggle_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Band list with checkboxes
        self.band_table = QTableWidget()
        self.band_table.setColumnCount(2)
        self.band_table.setHorizontalHeaderLabels(["Select", "Band Description"])
        self.band_table.horizontalHeader().setStretchLastSection(True)
        self.band_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.band_table.setRowCount(len(bands_info))
        
        for i, band_desc in enumerate(bands_info):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.band_table.setCellWidget(i, 0, checkbox_widget)
            
            # Band description (clickable)
            band_item = QTableWidgetItem(band_desc)
            band_item.setFlags(band_item.flags() & ~Qt.ItemIsEditable)
            self.band_table.setItem(i, 1, band_item)
        
        # Make entire row clickable
        self.band_table.cellClicked.connect(self.toggle_row_selection)
        layout.addWidget(self.band_table)
        
        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_selection)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def toggle_row_selection(self, row, col):
        """Toggle checkbox when clicking anywhere on the row"""
        checkbox_widget = self.band_table.cellWidget(row, 0)
        checkbox = checkbox_widget.findChild(QCheckBox)
        checkbox.setChecked(not checkbox.isChecked())
    
    def toggle_all(self):
        """Toggle all checkboxes"""
        all_selected = all(
            self.band_table.cellWidget(i, 0).findChild(QCheckBox).isChecked()
            for i in range(self.band_table.rowCount())
        )
        
        new_state = not all_selected
        for i in range(self.band_table.rowCount()):
            checkbox = self.band_table.cellWidget(i, 0).findChild(QCheckBox)
            checkbox.setChecked(new_state)
        
        self.toggle_button.setText("Unselect All" if new_state else "Select All")
    
    def accept_selection(self):
        """Get selected bands and close dialog"""
        self.selected_bands = []
        for i in range(self.band_table.rowCount()):
            checkbox = self.band_table.cellWidget(i, 0).findChild(QCheckBox)
            if checkbox.isChecked():
                self.selected_bands.append(i + 1)
        
        if not self.selected_bands:
            QMessageBox.warning(self, "No Bands Selected", "Please select at least one band.")
            return
        
        self.accept()


class LabelFieldSelectionDialog(QDialog):
    """Dialog for selecting label fields from reference shapefile"""
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Label Fields")
        self.resize(400, 300)
        self.fields = fields
        self.selected_fields = []
        
        layout = QVBoxLayout()
        
        # Select/Unselect All button
        button_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Select All")
        self.toggle_button.clicked.connect(self.toggle_all)
        button_layout.addWidget(self.toggle_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Field list with checkboxes
        self.field_table = QTableWidget()
        self.field_table.setColumnCount(2)
        self.field_table.setHorizontalHeaderLabels(["Select", "Field Name"])
        self.field_table.horizontalHeader().setStretchLastSection(True)
        self.field_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.field_table.setRowCount(len(fields))
        
        for i, field_name in enumerate(fields):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.field_table.setCellWidget(i, 0, checkbox_widget)
            
            # Field name (clickable)
            field_item = QTableWidgetItem(field_name)
            field_item.setFlags(field_item.flags() & ~Qt.ItemIsEditable)
            self.field_table.setItem(i, 1, field_item)
        
        # Make entire row clickable
        self.field_table.cellClicked.connect(self.toggle_row_selection)
        layout.addWidget(self.field_table)
        
        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_selection)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def toggle_row_selection(self, row, col):
        """Toggle checkbox when clicking anywhere on the row"""
        checkbox_widget = self.field_table.cellWidget(row, 0)
        checkbox = checkbox_widget.findChild(QCheckBox)
        checkbox.setChecked(not checkbox.isChecked())
    
    def toggle_all(self):
        """Toggle all checkboxes"""
        all_selected = all(
            self.field_table.cellWidget(i, 0).findChild(QCheckBox).isChecked()
            for i in range(self.field_table.rowCount())
        )
        
        new_state = not all_selected
        for i in range(self.field_table.rowCount()):
            checkbox = self.field_table.cellWidget(i, 0).findChild(QCheckBox)
            checkbox.setChecked(new_state)
        
        self.toggle_button.setText("Unselect All" if new_state else "Select All")
    
    def accept_selection(self):
        """Get selected fields and close dialog"""
        self.selected_fields = []
        for i in range(self.field_table.rowCount()):
            checkbox = self.field_table.cellWidget(i, 0).findChild(QCheckBox)
            if checkbox.isChecked():
                self.selected_fields.append(self.fields[i])
        
        if not self.selected_fields:
            QMessageBox.warning(self, "No Fields Selected", "Please select at least one field.")
            return
        
        self.accept()


class ClassificationDialog(QDialog):
    classify_signal = pyqtSignal(list, list, str, str, bool, bool, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Supervised Classification - Batch Processing")
        self.resize(1000, 700)
        
        self.raster_data = []
        self.reference_data = []
        
        layout = QVBoxLayout()

        # Pre-trained Model Checkbox
        self.use_pretrained_model = QCheckBox("Do you want to use a pre-trained model?")
        self.use_pretrained_model.setChecked(False)
        self.use_pretrained_model.stateChanged.connect(self.on_pretrained_model_checkbox_change)
        layout.addWidget(self.use_pretrained_model)
        
        # ========== INPUT RASTER TABLE ==========
        layout.addWidget(QLabel("Input Raster Layers:"))
        
        # Table for input rasters
        self.raster_table = QTableWidget()
        self.raster_table.setColumnCount(4)
        self.raster_table.setHorizontalHeaderLabels([
            "Select", "Raster Name", "Output File Name", "Selected Bands"
        ])
        self.raster_table.horizontalHeader().setStretchLastSection(False)
        self.raster_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.raster_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.raster_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.raster_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.raster_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.raster_table.cellClicked.connect(self.toggle_raster_row_selection)
        layout.addWidget(self.raster_table)
        
        # Buttons for input raster management
        raster_button_layout = QHBoxLayout()
        refresh_raster_button = QPushButton("Refresh Raster List")
        refresh_raster_button.clicked.connect(self.refresh_raster_list)
        add_raster_button = QPushButton("Add Raster from File")
        add_raster_button.clicked.connect(self.add_raster)
        self.select_all_raster_button = QPushButton("Select All")
        self.select_all_raster_button.clicked.connect(self.toggle_all_rasters)
        select_bands_button = QPushButton("Select Bands")
        select_bands_button.clicked.connect(self.select_bands_for_selected)
        remove_raster_button = QPushButton("Remove from List")
        remove_raster_button.clicked.connect(self.remove_raster)
        
        raster_button_layout.addWidget(refresh_raster_button)
        raster_button_layout.addWidget(add_raster_button)
        raster_button_layout.addWidget(self.select_all_raster_button)
        raster_button_layout.addWidget(select_bands_button)
        raster_button_layout.addWidget(remove_raster_button)
        raster_button_layout.addStretch()
        layout.addLayout(raster_button_layout)
        
        # ========== REFERENCE FILES TABLE ==========
        self.reference_label = QLabel("Reference Shapefiles:")
        layout.addWidget(self.reference_label)
        
        # Table for reference files
        self.reference_table = QTableWidget()
        self.reference_table.setColumnCount(3)
        self.reference_table.setHorizontalHeaderLabels([
            "Select", "Reference File Name", "Select Label Fields"
        ])
        self.reference_table.horizontalHeader().setStretchLastSection(False)
        self.reference_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.reference_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.reference_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.reference_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.reference_table.cellClicked.connect(self.toggle_reference_row_selection)
        layout.addWidget(self.reference_table)
        
        # Buttons for reference file management
        reference_button_layout = QHBoxLayout()
        self.refresh_reference_button = QPushButton("Refresh Reference List")
        self.refresh_reference_button.clicked.connect(self.refresh_reference_list)
        self.add_reference_button = QPushButton("Add Reference from File")
        self.add_reference_button.clicked.connect(self.add_reference)
        self.select_all_reference_button = QPushButton("Select All")
        self.select_all_reference_button.clicked.connect(self.toggle_all_references)
        self.remove_reference_button = QPushButton("Remove from List")
        self.remove_reference_button.clicked.connect(self.remove_reference)
        
        reference_button_layout.addWidget(self.refresh_reference_button)
        reference_button_layout.addWidget(self.add_reference_button)
        reference_button_layout.addWidget(self.select_all_reference_button)
        reference_button_layout.addWidget(self.remove_reference_button)
        reference_button_layout.addStretch()
        layout.addLayout(reference_button_layout)
        
        # ========== OUTPUT FOLDER SECTION ==========
        layout.addWidget(QLabel("Output Folder:"))
        output_layout = QHBoxLayout()
        self.output_folder = QLineEdit()
        output_browse_button = QPushButton("Browse")
        output_browse_button.clicked.connect(self.browse_folder)
        output_layout.addWidget(self.output_folder)
        output_layout.addWidget(output_browse_button)
        layout.addLayout(output_layout)
        
        self.save_same_as_input = QCheckBox("Save output in same folder as input?")
        self.save_same_as_input.stateChanged.connect(self.on_save_same_folder_change)
        layout.addWidget(self.save_same_as_input)
        
        # ========== CLASSIFICATION OPTIONS ==========
        layout.addWidget(QLabel("Classification Method:"))
        self.classification_methods = QComboBox()
        self.classification_methods.addItems([
            "Minimum Distance", "Random Forest", "SVM", "KNN (Not Good)", 
            "Maximum Likelihood (Not Working)"
        ])
        self.classification_methods.currentTextChanged.connect(self.on_method_change)
        layout.addWidget(self.classification_methods)
        
        self.save_model_checkbox = QCheckBox("Wants to save trained model?")
        self.save_model_checkbox.setChecked(True)
        self.save_model_checkbox.clicked.connect(self.on_save_model_checkbox_clicked)
        layout.addWidget(self.save_model_checkbox)
        
        layout.addWidget(QLabel("No of Iterations:"))
        self.num_iterations_spinbox = QSpinBox()
        self.num_iterations_spinbox.setMinimum(1)
        self.num_iterations_spinbox.setMaximum(1000)
        self.num_iterations_spinbox.setValue(100)
        layout.addWidget(self.num_iterations_spinbox)
        
        self.open_in_qgis = QCheckBox("Do you want to open output in QGIS Interface?")
        layout.addWidget(self.open_in_qgis)
        
        # ========== CLASSIFY BUTTON ==========
        classify_button = QPushButton("Classify")
        classify_button.clicked.connect(self.classify)
        layout.addWidget(classify_button)

        # ========== PROGRESS SECTION ==========
        self.processing_info_label = QLabel()
        self.processing_info_label.setWordWrap(True)
        layout.addWidget(self.processing_info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
        # Auto-load layers on initialization
        self.refresh_raster_list()
        self.refresh_reference_list()

    # ========== AUTO-REFRESH METHODS ==========
    def refresh_raster_list(self):
        """Automatically refresh raster list from QGIS project"""
        project = QgsProject.instance()
        layers = project.mapLayers().values()
        
        # Get existing raster paths to avoid duplicates
        existing_paths = {r['path'] for r in self.raster_data}
        
        for layer in layers:
            if layer.type() == QgsMapLayerType.RasterLayer:
                file_path = layer.source()
                
                # Skip if already in list
                if file_path in existing_paths:
                    continue
                
                # Get raster information
                ds = gdal.Open(file_path)
                if ds:
                    num_bands = ds.RasterCount
                    bands_info = [f"Band {i+1}" for i in range(num_bands)]
                    selected_bands = list(range(1, num_bands + 1))
                    
                    # Generate default output name
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_name = f"{base_name}_classified.tif"
                    
                    # Store raster data
                    self.raster_data.append({
                        'path': file_path,
                        'name': layer.name(),
                        'output_name': output_name,
                        'bands_info': bands_info,
                        'selected_bands': selected_bands,
                        'selected': True
                    })
                    
                    existing_paths.add(file_path)
                    ds = None
        
        self.update_raster_table()
    
    def refresh_reference_list(self):
        """Automatically refresh reference list from QGIS project"""
        if self.use_pretrained_model.isChecked():
            return  # Don't auto-load for pre-trained models
        
        project = QgsProject.instance()
        layers = project.mapLayers().values()
        
        # Get existing reference paths to avoid duplicates
        existing_paths = {r['path'] for r in self.reference_data}
        
        for layer in layers:
            if layer.type() == QgsMapLayerType.VectorLayer:
                file_path = layer.source().split('|')[0]  # Remove any layer subsets
                
                # Skip if already in list
                if file_path in existing_paths:
                    continue
                
                # Get fields from vector layer
                fields = [field.name() for field in layer.fields()]
                
                # Store reference data
                self.reference_data.append({
                    'path': file_path,
                    'name': layer.name(),
                    'fields': fields,
                    'selected_fields': [],
                    'selected': True
                })
                
                existing_paths.add(file_path)
        
        self.update_reference_table()

    # ========== RASTER TABLE METHODS ==========
    def add_raster(self):
        """Add raster file(s) to the table"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Raster Images", "", "Images (*.tif *.tiff *.img *.jpg *.png)"
        )
        
        existing_paths = {r['path'] for r in self.raster_data}
        
        for file_path in file_paths:
            if file_path and file_path not in existing_paths:
                # Get raster information
                ds = gdal.Open(file_path)
                if ds:
                    num_bands = ds.RasterCount
                    bands_info = [f"Band {i+1}" for i in range(num_bands)]
                    selected_bands = list(range(1, num_bands + 1))
                    
                    # Generate default output name
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_name = f"{base_name}_classified.tif"
                    
                    # Store raster data
                    self.raster_data.append({
                        'path': file_path,
                        'name': os.path.basename(file_path),
                        'output_name': output_name,
                        'bands_info': bands_info,
                        'selected_bands': selected_bands,
                        'selected': True
                    })
                    
                    ds = None
        
        self.update_raster_table()
    
    def update_raster_table(self):
        """Update the raster table display"""
        self.raster_table.setRowCount(len(self.raster_data))
        
        for i, raster in enumerate(self.raster_data):
            # Selection checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(raster['selected'])
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.raster_table.setCellWidget(i, 0, checkbox_widget)
            
            # Raster name
            name_item = QTableWidgetItem(raster['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.raster_table.setItem(i, 1, name_item)
            
            # Output file name (editable)
            output_item = QTableWidgetItem(raster['output_name'])
            self.raster_table.setItem(i, 2, output_item)
            
            # Selected bands with "..." button
            bands_text = f"{len(raster['selected_bands'])} bands"
            bands_button = QPushButton(f"{bands_text} ...")
            bands_button.clicked.connect(lambda checked, row=i: self.select_bands(row))
            self.raster_table.setCellWidget(i, 3, bands_button)
    
    def toggle_raster_row_selection(self, row, col):
        """Toggle raster selection when clicking anywhere on the row (except buttons and output name)"""
        # Don't toggle when clicking on:
        # - col 2: Output File Name (editable field)
        # - col 3: Selected Bands button
        if col not in [2, 3]:
            checkbox_widget = self.raster_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            checkbox.setChecked(not checkbox.isChecked())
            self.raster_data[row]['selected'] = checkbox.isChecked()
    
    def toggle_all_rasters(self):
        """Toggle all raster selections"""
        all_selected = all(r['selected'] for r in self.raster_data)
        new_state = not all_selected
        
        for raster in self.raster_data:
            raster['selected'] = new_state
        
        self.update_raster_table()
        self.select_all_raster_button.setText("Unselect All" if new_state else "Select All")
    
    def select_bands(self, row):
        """Open band selection dialog for a specific raster"""
        raster = self.raster_data[row]
        dialog = BandSelectionDialog(raster['bands_info'], self)
        
        # Pre-select previously selected bands
        for i, checkbox_widget in enumerate([dialog.band_table.cellWidget(i, 0) for i in range(len(raster['bands_info']))]):
            checkbox = checkbox_widget.findChild(QCheckBox)
            checkbox.setChecked((i + 1) in raster['selected_bands'])
        
        if dialog.exec_() == QDialog.Accepted:
            raster['selected_bands'] = dialog.selected_bands
            self.update_raster_table()
    
    def select_bands_for_selected(self):
        """Select bands for all selected rasters"""
        selected_rows = [i for i, r in enumerate(self.raster_data) if r['selected']]
        
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one raster.")
            return
        
        for row in selected_rows:
            self.select_bands(row)
    
    def remove_raster(self):
        """Remove selected rasters from the list"""
        to_remove = [i for i, r in enumerate(self.raster_data) if r['selected']]
        
        if not to_remove:
            QMessageBox.warning(self, "No Selection", "Please select rasters to remove.")
            return
        
        for i in reversed(to_remove):
            del self.raster_data[i]
        
        self.update_raster_table()
    
    # ========== REFERENCE TABLE METHODS ==========
    def add_reference(self):
        """Add reference file(s) to the table"""
        if self.use_pretrained_model.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Model Files", "", "Pickle Files (*.pkl)"
            )
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Shapefiles", "", "Shapefiles (*.shp)"
            )
        
        existing_paths = {r['path'] for r in self.reference_data}
        
        for file_path in file_paths:
            if file_path and file_path not in existing_paths:
                fields = []
                if not self.use_pretrained_model.isChecked():
                    # Get fields from shapefile
                    shapefile_ds = ogr.Open(file_path)
                    if shapefile_ds:
                        layer = shapefile_ds.GetLayer()
                        if layer:
                            layer_def = layer.GetLayerDefn()
                            fields = [layer_def.GetFieldDefn(i).GetName() 
                                    for i in range(layer_def.GetFieldCount())]
                
                # Store reference data
                self.reference_data.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'fields': fields,
                    'selected_fields': [],
                    'selected': True
                })
        
        self.update_reference_table()
    
    def update_reference_table(self):
        """Update the reference table display"""
        self.reference_table.setRowCount(len(self.reference_data))
        
        is_model_mode = self.use_pretrained_model.isChecked()
        
        for i, reference in enumerate(self.reference_data):
            # Selection checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(reference['selected'])
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.reference_table.setCellWidget(i, 0, checkbox_widget)
            
            # Reference/Model name
            name_item = QTableWidgetItem(reference['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.reference_table.setItem(i, 1, name_item)
            
            # Label fields with "..." button (only in reference mode)
            if not is_model_mode:
                fields_text = f"{len(reference['selected_fields'])} fields" if reference['selected_fields'] else "Select..."
                fields_button = QPushButton(f"{fields_text} ...")
                fields_button.clicked.connect(lambda checked, row=i: self.select_label_fields(row))
                self.reference_table.setCellWidget(i, 2, fields_button)
    
    def toggle_reference_row_selection(self, row, col):
        """Toggle reference selection when clicking anywhere on the row"""
        is_model_mode = self.use_pretrained_model.isChecked()
        
        # In model mode (2 columns): toggle on name column (col 1)
        # In reference mode (3 columns): don't toggle when clicking on fields button (col 2)
        if is_model_mode:
            # Model mode - only 2 columns, toggle on name column (col 1)
            if col == 1:
                checkbox_widget = self.reference_table.cellWidget(row, 0)
                checkbox = checkbox_widget.findChild(QCheckBox)
                checkbox.setChecked(not checkbox.isChecked())
                self.reference_data[row]['selected'] = checkbox.isChecked()
        else:
            # Reference mode - 3 columns, don't toggle on fields button (col 2)
            if col != 2:
                checkbox_widget = self.reference_table.cellWidget(row, 0)
                checkbox = checkbox_widget.findChild(QCheckBox)
                checkbox.setChecked(not checkbox.isChecked())
                self.reference_data[row]['selected'] = checkbox.isChecked()
    
    def toggle_all_references(self):
        """Toggle all reference selections"""
        all_selected = all(r['selected'] for r in self.reference_data)
        new_state = not all_selected
        
        for reference in self.reference_data:
            reference['selected'] = new_state
        
        self.update_reference_table()
        self.select_all_reference_button.setText("Unselect All" if new_state else "Select All")
    
    def select_label_fields(self, row):
        """Open label field selection dialog"""
        reference = self.reference_data[row]
        
        if not reference['fields']:
            QMessageBox.warning(self, "No Fields", "No fields available in this shapefile.")
            return
        
        dialog = LabelFieldSelectionDialog(reference['fields'], self)
        
        # Pre-select previously selected fields
        for i, checkbox_widget in enumerate([dialog.field_table.cellWidget(i, 0) for i in range(len(reference['fields']))]):
            checkbox = checkbox_widget.findChild(QCheckBox)
            checkbox.setChecked(reference['fields'][i] in reference['selected_fields'])
        
        if dialog.exec_() == QDialog.Accepted:
            reference['selected_fields'] = dialog.selected_fields
            self.update_reference_table()
    
    def remove_reference(self):
        """Remove selected references from the list"""
        to_remove = [i for i, r in enumerate(self.reference_data) if r['selected']]
        
        if not to_remove:
            QMessageBox.warning(self, "No Selection", "Please select references to remove.")
            return
        
        for i in reversed(to_remove):
            del self.reference_data[i]
        
        self.update_reference_table()
    
    # ========== OTHER METHODS ==========
    def browse_folder(self):
        """Browse for output folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder.setText(folder_path)
    
    def on_save_same_folder_change(self, state):
        """Handle save same folder checkbox"""
        if state == Qt.Checked:
            self.output_folder.setEnabled(False)
        else:
            self.output_folder.setEnabled(True)
    
    def on_pretrained_model_checkbox_change(self, state):
        """Handle pre-trained model checkbox change"""
        if state == Qt.Checked:
            # Change to Model mode
            self.reference_label.setText("Model Files:")
            
            # Change table to 2 columns (remove Select Label Fields)
            self.reference_table.setColumnCount(2)
            self.reference_table.setHorizontalHeaderLabels([
                "Select", "Model File Name"
            ])
            self.reference_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.reference_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            
            # Change button labels and visibility
            self.refresh_reference_button.setVisible(False)  # Hide refresh button
            self.add_reference_button.setText("Add Model from File")
            
            # Disable classification options
            self.classification_methods.setEnabled(False)
            self.save_model_checkbox.setChecked(False)
            self.save_model_checkbox.setEnabled(False)
        else:
            # Change to Reference mode
            self.reference_label.setText("Reference Shapefiles:")
            
            # Change table to 3 columns (add Select Label Fields)
            self.reference_table.setColumnCount(3)
            self.reference_table.setHorizontalHeaderLabels([
                "Select", "Reference File Name", "Select Label Fields"
            ])
            self.reference_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.reference_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.reference_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            
            # Change button labels and visibility
            self.refresh_reference_button.setVisible(True)  # Show refresh button
            self.add_reference_button.setText("Add Reference from File")
            
            # Enable classification options
            self.classification_methods.setEnabled(True)
            self.save_model_checkbox.setEnabled(True)
            self.save_model_checkbox.setChecked(True)
        
        # Clear reference table when switching
        self.reference_data.clear()
        self.update_reference_table()
        
        # Refresh if switching back to shapefile mode
        if state != Qt.Checked:
            self.refresh_reference_list()
    
    def on_method_change(self):
        """Handle classification method change"""
        method = self.classification_methods.currentText()
        if method in ["Minimum Distance", "Maximum Likelihood (Not Working)"]:
            self.save_model_checkbox.setChecked(False)
            self.save_model_checkbox.setEnabled(False)
            self.num_iterations_spinbox.setVisible(False)
        else:
            self.save_model_checkbox.setChecked(True)
            self.save_model_checkbox.setEnabled(True)
            self.num_iterations_spinbox.setVisible(True)
    
    def on_save_model_checkbox_clicked(self):
        """Handle save model checkbox click"""
        if not self.save_model_checkbox.isEnabled():
            method = self.classification_methods.currentText()
            QMessageBox.warning(
                self, "Unsupported Option", 
                f"Your selected model {method} does not support this option."
            )
            self.save_model_checkbox.setChecked(False)
    
    def classify(self):
        """Start classification process"""
        # Update output names from table
        for i in range(self.raster_table.rowCount()):
            output_item = self.raster_table.item(i, 2)
            if output_item:
                self.raster_data[i]['output_name'] = output_item.text()
        
        # Get selected rasters
        selected_rasters = [r for r in self.raster_data if r['selected']]
        if not selected_rasters:
            QMessageBox.critical(self, "No Rasters", "Please select at least one raster.")
            return
        
        # Get selected references
        selected_references = [r for r in self.reference_data if r['selected']]
        if not selected_references:
            QMessageBox.critical(self, "No References", "Please select at least one reference file.")
            return
        
        # Check label fields for non-pretrained mode
        if not self.use_pretrained_model.isChecked():
            for ref in selected_references:
                if not ref['selected_fields']:
                    QMessageBox.critical(
                        self, "Missing Label Fields", 
                        f"Please select label fields for {ref['name']}"
                    )
                    return
        
        # Get output folder
        output_folder = self.output_folder.text()
        if not self.save_same_as_input.isChecked() and not output_folder:
            QMessageBox.critical(self, "Missing Output", "Please provide output folder.")
            return
        
        method = self.classification_methods.currentText()
        open_in_qgis = self.open_in_qgis.isChecked()
        save_model = self.save_model_checkbox.isChecked()
        num_iterations = self.num_iterations_spinbox.value()
        use_pretrained_model = self.use_pretrained_model.isChecked()
        
        self.progress_bar.setValue(0)
        self.processing_info_label.setText("Classification process started...")
        
        # Emit signal with all necessary data
        self.classify_signal.emit(
            selected_rasters, 
            selected_references, 
            output_folder, 
            method, 
            open_in_qgis, 
            save_model, 
            num_iterations, 
            use_pretrained_model
        )
    
    def show_processing_info(self, info):
        """Show processing information"""
        self.processing_info_label.setText(info)
