"""
Robust Dependency Installer for QGIS Plugins
Correctly handles Python executable detection on Windows (OSGeo4W)

Author: MAS Tools
"""

import sys
import os
import subprocess
from pathlib import Path

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import (
    QMessageBox, QProgressDialog, QApplication, QDialog,
    QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
)
from qgis.PyQt.QtCore import Qt


class DependencyInstaller:
    """Handle installation of Python dependencies for QGIS plugins"""
    
    PLUGIN_NAME = "Supervised Classifier"
    
    def __init__(self, iface, required_packages: dict):
        """
        Initialize the installer.
        
        Args:
            iface: QGIS interface
            required_packages: Dict of {pip_package_name: import_module_name}
                e.g., {'scikit-learn': 'sklearn', 'scipy': 'scipy'}
        """
        self.iface = iface
        self.required_packages = required_packages
        self._python_exe = None
    
    def get_python_executable(self) -> str:
        """
        Get the actual Python executable path.
        
        On Windows QGIS, sys.executable points to qgis.exe, not python.exe.
        This method finds the correct Python interpreter.
        """
        if self._python_exe:
            return self._python_exe
        
        if os.name == 'nt':  # Windows
            self._python_exe = self._find_windows_python()
        else:  # Linux/macOS
            self._python_exe = sys.executable
        
        self._log(f"Python executable: {self._python_exe}")
        return self._python_exe
    
    def _find_windows_python(self) -> str:
        """Find Python executable on Windows QGIS installation"""
        
        self._log(f"sys.executable = {sys.executable}")
        self._log(f"OSGEO4W_ROOT = {os.environ.get('OSGEO4W_ROOT', 'NOT SET')}")
        
        # Method 1: Check OSGEO4W_ROOT environment variable
        osgeo4w_root = os.environ.get('OSGEO4W_ROOT')
        if osgeo4w_root:
            self._log(f"Checking OSGEO4W_ROOT: {osgeo4w_root}")
            # Try different Python versions (newest first)
            for py_ver in ['Python312', 'Python311', 'Python310', 'Python39', 'Python38', 'Python37']:
                python_path = os.path.join(osgeo4w_root, 'apps', py_ver, 'python.exe')
                self._log(f"  Checking {python_path}: {os.path.exists(python_path)}")
                if os.path.exists(python_path):
                    self._log(f"Found Python via OSGEO4W_ROOT: {python_path}")
                    return python_path
        
        # Method 2: Check PATH for QGIS Python
        self._log("Checking PATH for QGIS Python...")
        for path_dir in os.environ.get('PATH', '').split(os.pathsep):
            if 'QGIS' in path_dir.upper() or 'OSGEO4W' in path_dir.upper():
                python_path = os.path.join(path_dir, 'python.exe')
                if os.path.exists(python_path):
                    self._log(f"Found Python in PATH: {python_path}")
                    return python_path
        
        # Method 3: Try to find from QGIS installation directory
        # sys.executable on QGIS points to qgis.exe, we can navigate from there
        qgis_dir = os.path.dirname(sys.executable)
        self._log(f"QGIS directory: {qgis_dir}")
        
        possible_paths = [
            os.path.join(qgis_dir, '..', 'apps', 'Python312', 'python.exe'),
            os.path.join(qgis_dir, '..', 'apps', 'Python311', 'python.exe'),
            os.path.join(qgis_dir, '..', 'apps', 'Python310', 'python.exe'),
            os.path.join(qgis_dir, '..', 'apps', 'Python39', 'python.exe'),
            os.path.join(qgis_dir, '..', 'apps', 'Python38', 'python.exe'),
            os.path.join(qgis_dir, 'python.exe'),
        ]
        
        for path in possible_paths:
            normalized = os.path.normpath(path)
            self._log(f"  Checking {normalized}: {os.path.exists(normalized)}")
            if os.path.exists(normalized):
                self._log(f"Found Python via QGIS dir: {normalized}")
                return normalized
        
        # Method 4: Use python-qgis.bat wrapper if available
        python_qgis_bat = os.path.join(qgis_dir, 'python-qgis.bat')
        if os.path.exists(python_qgis_bat):
            self._log(f"Found python-qgis.bat: {python_qgis_bat}")
            return python_qgis_bat
        
        # Method 5: Look for python-qgis-ltr.bat 
        python_qgis_ltr_bat = os.path.join(qgis_dir, 'python-qgis-ltr.bat')
        if os.path.exists(python_qgis_ltr_bat):
            self._log(f"Found python-qgis-ltr.bat: {python_qgis_ltr_bat}")
            return python_qgis_ltr_bat
        
        # Fallback: This will likely fail on QGIS Windows, but log it
        self._log("WARNING: Could not find Python executable, falling back to sys.executable", Qgis.Warning)
        return sys.executable
    
    def show_debug_info(self):
        """Show diagnostic information about Python detection"""
        python_exe = self.get_python_executable()
        
        info_lines = [
            f"sys.executable: {sys.executable}",
            f"sys.prefix: {sys.prefix}",
            f"sys.path[0]: {sys.path[0] if sys.path else 'N/A'}",
            f"",
            f"OSGEO4W_ROOT: {os.environ.get('OSGEO4W_ROOT', 'NOT SET')}",
            f"PYTHONHOME: {os.environ.get('PYTHONHOME', 'NOT SET')}",
            f"",
            f"Detected Python: {python_exe}",
            f"Python exists: {os.path.exists(python_exe)}",
        ]
        
        # Test pip
        if os.path.exists(python_exe):
            try:
                result = subprocess.run(
                    [python_exe, '-m', 'pip', '--version'],
                    capture_output=True, text=True, timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                info_lines.append(f"\nPip version: {result.stdout.strip()}")
                if result.stderr:
                    info_lines.append(f"Pip stderr: {result.stderr.strip()}")
            except Exception as e:
                info_lines.append(f"\nPip check failed: {e}")
        
        # Show in message box
        QMessageBox.information(
            self.iface.mainWindow() if self.iface else None,
            "Dependency Installer Debug Info",
            "\n".join(info_lines)
        )
        
        return "\n".join(info_lines)

    def _log(self, message: str, level=Qgis.Info):
        """Log message to QGIS message log"""
        QgsMessageLog.logMessage(message, self.PLUGIN_NAME, level)
        # Also print to console for debugging
        print(f"[{self.PLUGIN_NAME}] {message}")
    
    def check_dependencies(self) -> list:
        """
        Check which required packages are missing.
        
        Returns:
            List of missing package names (pip names)
        """
        missing = []
        
        for package_name, import_name in self.required_packages.items():
            try:
                __import__(import_name)
                self._log(f"✓ {package_name} is installed")
            except ImportError:
                missing.append(package_name)
                self._log(f"✗ {package_name} is missing", Qgis.Warning)
        
        return missing
    
    def check_and_install(self, silent_if_ok: bool = True) -> bool:
        """
        Check dependencies and prompt to install if missing.
        
        Args:
            silent_if_ok: If True, don't show message if all deps are installed
            
        Returns:
            True if all dependencies are available, False otherwise
        """
        missing = self.check_dependencies()
        
        if not missing:
            if not silent_if_ok:
                self._log("All dependencies are installed", Qgis.Success)
            return True
        
        # Show installation dialog
        return self._show_install_dialog(missing)
    
    def _show_install_dialog(self, missing_packages: list) -> bool:
        """Show dialog to install missing packages"""
        
        dialog = DependencyInstallDialog(
            self.iface.mainWindow(),
            missing_packages,
            self.PLUGIN_NAME
        )
        
        if dialog.exec_() != QDialog.Accepted:
            return False
        
        # Perform installation
        return self._install_packages(missing_packages)
    
    def _install_packages(self, packages: list) -> bool:
        """Install packages using pip"""
        
        python_exe = self.get_python_executable()
        
        # Verify Python executable exists
        if not os.path.exists(python_exe):
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Installation Error",
                f"Python executable not found at:\n{python_exe}\n\n"
                "Please install dependencies manually using OSGeo4W Shell:\n"
                f"pip install {' '.join(packages)}"
            )
            return False
        
        # Create progress dialog
        progress = QProgressDialog(
            "Installing dependencies...",
            "Cancel",
            0, len(packages),
            self.iface.mainWindow()
        )
        progress.setWindowTitle("Installing Dependencies")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.show()
        
        success_count = 0
        failed_packages = []
        error_messages = []
        
        for i, package in enumerate(packages):
            if progress.wasCanceled():
                break
            
            progress.setValue(i)
            progress.setLabelText(f"Installing {package}...")
            QApplication.processEvents()
            
            success, error = self._install_single_package(python_exe, package)
            if success:
                success_count += 1
                self._log(f"✓ Successfully installed {package}", Qgis.Success)
            else:
                failed_packages.append(package)
                error_messages.append(f"{package}: {error}")
                self._log(f"✗ Failed to install {package}: {error}", Qgis.Critical)
        
        progress.setValue(len(packages))
        progress.close()
        
        # Show results
        if failed_packages:
            self._show_failure_dialog(failed_packages, error_messages)
            return False
        else:
            self._show_success_dialog(success_count)
            return True
    
    def _install_single_package(self, python_exe: str, package: str) -> tuple:
        """
        Install a single package.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            cmd = [
                python_exe,
                '-m', 'pip',
                'install',
                '--user',
                '--upgrade',
                package
            ]
            
            self._log(f"Running: {' '.join(cmd)}")
            
            # Use CREATE_NO_WINDOW on Windows to prevent console popup
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                creationflags=creationflags
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr or result.stdout or "Unknown error"
                
        except subprocess.TimeoutExpired:
            return False, "Installation timed out (5 minutes)"
        except FileNotFoundError:
            return False, f"Python executable not found: {python_exe}"
        except Exception as e:
            return False, str(e)
    
    def _show_success_dialog(self, count: int):
        """Show success message"""
        QMessageBox.information(
            self.iface.mainWindow(),
            "Installation Complete",
            f"Successfully installed {count} package(s)!\n\n"
            "⚠️ Please restart QGIS to use the plugin."
        )
    
    def _show_failure_dialog(self, failed: list, errors: list):
        """Show detailed failure dialog"""
        dialog = QDialog(self.iface.mainWindow())
        dialog.setWindowTitle("Installation Failed")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # Header
        header = QLabel(f"Failed to install {len(failed)} package(s):")
        header.setStyleSheet("font-weight: bold; color: #c0392b;")
        layout.addWidget(header)
        
        # Error details
        error_text = QTextEdit()
        error_text.setReadOnly(True)
        error_text.setPlainText("\n\n".join(errors))
        error_text.setMaximumHeight(150)
        layout.addWidget(error_text)
        
        # Manual install instructions
        manual_label = QLabel(
            "<b>To install manually:</b><br>"
            "1. Open <b>OSGeo4W Shell</b> from Start Menu<br>"
            "2. Run the following command:"
        )
        layout.addWidget(manual_label)
        
        cmd_text = QTextEdit()
        cmd_text.setReadOnly(True)
        cmd_text.setPlainText(f"pip install --user {' '.join(failed)}")
        cmd_text.setMaximumHeight(40)
        cmd_text.setStyleSheet("font-family: Consolas, monospace; background: #f5f5f5;")
        layout.addWidget(cmd_text)
        
        # OK button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()


class DependencyInstallDialog(QDialog):
    """Dialog to confirm dependency installation"""
    
    def __init__(self, parent, packages: list, plugin_name: str):
        super().__init__(parent)
        self.setWindowTitle(f"{plugin_name} - Missing Dependencies")
        self.setMinimumWidth(450)
        
        layout = QVBoxLayout(self)
        
        # Info icon and message
        header = QLabel(
            f"<h3>📦 Missing Dependencies</h3>"
            f"<p>The <b>{plugin_name}</b> plugin requires the following "
            f"Python packages to be installed:</p>"
        )
        layout.addWidget(header)
        
        # Package list
        packages_html = "<ul>" + "".join(f"<li><code>{pkg}</code></li>" for pkg in packages) + "</ul>"
        pkg_label = QLabel(packages_html)
        pkg_label.setStyleSheet("margin-left: 20px;")
        layout.addWidget(pkg_label)
        
        # Note
        note = QLabel(
            "<p><small>This will install packages to your user directory using pip.<br>"
            "You will need to restart QGIS after installation.</small></p>"
        )
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        install_btn = QPushButton("Install Now")
        install_btn.setDefault(True)
        install_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #2980b9; }"
        )
        install_btn.clicked.connect(self.accept)
        btn_layout.addWidget(install_btn)
        
        layout.addLayout(btn_layout)


def check_and_install_dependencies(iface, required_packages: dict, plugin_name: str = "QGIS Plugin") -> bool:
    """
    Convenience function to check and install dependencies.
    
    Args:
        iface: QGIS interface
        required_packages: Dict of {pip_package_name: import_module_name}
        plugin_name: Name of the plugin for display
        
    Returns:
        True if all dependencies are available, False otherwise
    """
    installer = DependencyInstaller(iface, required_packages)
    installer.PLUGIN_NAME = plugin_name
    return installer.check_and_install()
