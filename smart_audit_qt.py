import os
import sys
import csv
from PySide2.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QComboBox, QLabel, QMessageBox, QSplitter, QTextEdit
)
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWebEngineWidgets import QWebEngineView
from comprehensive_audit_engine import ComprehensiveAuditEngine

def is_business_file(filename):
    ignore_ext = ['.log', '.pyc', '.tmp', '.DS_Store']
    if any(filename.endswith(ext) for ext in ignore_ext):
        return False
    if filename.startswith('.'):
        return False
    return True

def scan_files(root_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'file_organizer_cache']]
        for fname in filenames:
            if is_business_file(fname):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                files.append({
                    'filename': fname,
                    'full_path': full_path,
                    'rel_path': rel_path,
                    'folder': os.path.relpath(dirpath, root_dir)
                })
    return files

def get_all_folders(root_dir):
    folders = set()
    for dirpath, dirnames, _ in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'file_organizer_cache']]
        for d in dirnames:
            folders.add(os.path.relpath(os.path.join(dirpath, d), root_dir))
    folders.add('.')
    return sorted(folders)

def run_audit(target_dir, files):
    config = {'enable_ai': True, 'zero_change_mode': True}
    engine = ComprehensiveAuditEngine(config)
    audit_results = engine.run_comprehensive_audit(target_dir)
    analyzed = audit_results.get('files_analyzed', [])
    file_map = {f['full_path']: f for f in analyzed}
    result = []
    for f in files:
        audit_info = file_map.get(f['full_path'], {})
        system_suggestion = audit_info.get('Suggested_New_Path') or audit_info.get('document_type') or audit_info.get('Document_Type') or ''
        needs_manual = (
            audit_info.get('Requires_User_Decision') or
            audit_info.get('Document_Type') in ['Unclassified', 'REVIEW_NEEDED', 'REVIEW_CLASSIFY']
        )
        result.append({
            'filename': f['filename'],
            'full_path': f['full_path'],
            'folder': f['folder'],
            'system_suggestion': system_suggestion,
            'selected_folder': f['folder'],
            'needs_manual': needs_manual
        })
    return result

class AuditApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Smart Audit Qt')
        self.resize(1600, 1000)
        self.root_dir = ''
        self.files = []
        self.folders = []
        self.audit_data = []
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet('QSplitter::handle { background: #e0e0e0; }')
        
        # Left side - Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Όνομα', 'Επιλογή φακέλου', 'Τρέχων φάκελος', 'Πρόταση συστήματος'])
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnWidth(1, 220)
        self.tree.setColumnWidth(2, 180)
        self.tree.setColumnWidth(3, 220)
        self.tree.setStyleSheet('font-size: 15px; font-family: Inter, Arial, sans-serif;')
        self.tree.header().setStyleSheet('font-weight: bold; font-size: 16px; background: #f5f5f5;')
        self.tree.header().setSectionResizeMode(0, self.tree.header().ResizeToContents)
        self.tree.header().setSectionResizeMode(1, self.tree.header().ResizeToContents)
        self.tree.header().setSectionResizeMode(2, self.tree.header().ResizeToContents)
        self.tree.header().setSectionResizeMode(3, self.tree.header().ResizeToContents)
        self.tree.itemClicked.connect(self.on_file_selected)
        
        # Right side - Preview widget
        self.preview_widget = QWidget()
        self.preview_layout = QVBoxLayout()
        self.preview_label = QLabel('Επιλέξτε ένα αρχείο για προεπισκόπηση')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet('font-size: 18px; color: #888;')
        self.preview_layout.addWidget(self.preview_label)
        self.preview_widget.setLayout(self.preview_layout)
        self.preview_widget.setStyleSheet('background: #fff; border-radius: 12px; padding: 24px;')
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.tree)
        self.main_splitter.addWidget(self.preview_widget)
        self.main_splitter.setSizes([900, 700])
        
        # Buttons and controls
        self.export_btn = QPushButton('Export Επιλογών')
        self.export_btn.setStyleSheet('font-size: 16px; padding: 10px 30px; background: #1976d2; color: white; border-radius: 8px;')
        self.export_btn.clicked.connect(self.export_choices)
        self.export_btn.setEnabled(False)
        self.folder_label = QLabel('Επιλογή φακέλου εργασίας:')
        self.folder_label.setStyleSheet('font-size: 15px;')
        self.folder_btn = QPushButton('Browse')
        self.folder_btn.setStyleSheet('font-size: 15px;')
        self.folder_btn.clicked.connect(self.select_folder)
        self.folder_path_label = QLabel('')
        self.folder_path_label.setStyleSheet('font-size: 15px; color: #1976d2;')
        self.scan_btn = QPushButton('Σάρωση & Audit')
        self.scan_btn.setStyleSheet('font-size: 15px; background: #43a047; color: white; border-radius: 8px; padding: 6px 18px;')
        self.scan_btn.clicked.connect(self.scan_and_audit)
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('font-size: 15px; color: #555;')
        
        # Layout setup
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(12)
        top_layout.addWidget(self.folder_label)
        top_layout.addWidget(self.folder_btn)
        top_layout.addWidget(self.folder_path_label)
        top_layout.addWidget(self.scan_btn)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.main_splitter, stretch=1)
        main_layout.addWidget(self.export_btn, alignment=Qt.AlignRight)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Preview widgets
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet('font-size: 15px; background: #fafafa; border-radius: 8px; padding: 12px;')
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet('background: #fafafa; border-radius: 8px;')
        self.pdf_preview = QWebEngineView()
        self.pdf_preview.setStyleSheet('background: #fafafa; border-radius: 8px;')
        
        # Hide all preview widgets initially
        self.text_preview.hide()
        self.image_preview.hide()
        self.pdf_preview.hide()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Επιλογή φακέλου εργασίας')
        if folder:
            self.root_dir = folder
            self.folder_path_label.setText(folder)

    def scan_and_audit(self):
        if not self.root_dir or not os.path.isdir(self.root_dir):
            QMessageBox.warning(self, 'Σφάλμα', 'Ο φάκελος δεν υπάρχει!')
            return
        self.status_label.setText('Σάρωση και ανάλυση...')
        QApplication.processEvents()
        self.files = scan_files(self.root_dir)
        self.folders = get_all_folders(self.root_dir)
        self.audit_data = run_audit(self.root_dir, self.files)
        self.populate_tree()
        self.export_btn.setEnabled(True)
        self.status_label.setText(f'Βρέθηκαν {len(self.audit_data)} αρχεία.')

    def populate_tree(self):
        self.tree.clear()
        folder_map = {}
        for f in self.audit_data:
            folder_map.setdefault(f['folder'], []).append(f)
        for folder in sorted(folder_map.keys()):
            folder_item = QTreeWidgetItem([folder, '', '', ''])
            self.tree.addTopLevelItem(folder_item)
            for file_info in folder_map[folder]:
                file_item = QTreeWidgetItem([
                    file_info['filename'],
                    '',  # Drop-down will be set here
                    file_info['folder'],
                    file_info['system_suggestion']
                ])
                folder_item.addChild(file_item)
                # Add combo box for all files in the 2nd column
                combo = QComboBox()
                combo.addItems(self.folders)
                combo.setCurrentText(file_info['folder'])
                combo.currentTextChanged.connect(lambda val, fi=file_info: self.set_manual_folder(fi, val))
                self.tree.setItemWidget(file_item, 1, combo)
            folder_item.setExpanded(False)

    def set_manual_folder(self, file_info, folder):
        file_info['selected_folder'] = folder

    def export_choices(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Αποθήκευση επιλογών ως CSV', '', 'CSV Files (*.csv)')
        if path:
            with open(path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['filename', 'full_path', 'selected_folder', 'system_suggestion'])
                writer.writeheader()
                for row in self.audit_data:
                    writer.writerow({
                        'filename': row['filename'],
                        'full_path': row['full_path'],
                        'selected_folder': row.get('selected_folder', row['folder']),
                        'system_suggestion': row['system_suggestion']
                    })
            QMessageBox.information(self, 'Επιτυχία', 'Επιτυχής εξαγωγή!')

    def on_file_selected(self, item, column):
        if not item.parent():  # Skip folder items
            return
            
        # Find the file info
        file_info = None
        for f in self.audit_data:
            if f['filename'] == item.text(0):
                file_info = f
                break
                
        if not file_info:
            return
            
        # Clear previous preview
        self.text_preview.hide()
        self.image_preview.hide()
        self.pdf_preview.hide()
        self.preview_label.hide()
        
        # Show appropriate preview
        file_path = file_info['full_path']
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.txt', '.py', '.js', '.html', '.css', '.json', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text_preview.setText(content)
                self.text_preview.show()
            except:
                self.preview_label.setText('Δεν ήταν δυνατή η προεπισκόπηση του αρχείου')
                self.preview_label.show()
                
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_preview.setPixmap(scaled_pixmap)
                self.image_preview.show()
            except:
                self.preview_label.setText('Δεν ήταν δυνατή η προεπισκόπηση της εικόνας')
                self.preview_label.show()
                
        elif ext == '.pdf':
            try:
                self.pdf_preview.setUrl(QUrl.fromLocalFile(file_path))
                self.pdf_preview.show()
            except:
                self.preview_label.setText('Δεν ήταν δυνατή η προεπισκόπηση του PDF')
                self.preview_label.show()
        else:
            self.preview_label.setText('Δεν υποστηρίζεται προεπισκόπηση για αυτόν τον τύπο αρχείου')
            self.preview_label.show()

def main():
    app = QApplication(sys.argv)
    win = AuditApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()