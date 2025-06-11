#!/usr/bin/env python3
"""
🎛️ COMPREHENSIVE AUDIT UI
Interactive UI for file auditing and backup management
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import json
import datetime
import webbrowser
import csv
from feedback_trainer import FeedbackTrainer
from comprehensive_audit_engine import ComprehensiveAuditEngine
from excel_report_generator import ExcelReportGenerator
from icloud_backup_manager import iCloudBackupManager, create_progress_callback
from config_manager import ConfigManager

try:
    from comprehensive_audit_engine import ComprehensiveAuditEngine
    from excel_report_generator import ExcelReportGenerator
    from icloud_backup_manager import iCloudBackupManager, create_progress_callback
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    MODULES_AVAILABLE = False

class AuditUI:
    def __init__(self, root):
        print("[DEBUG] AuditUI.__init__ started")
        self.root = root
        self.root.title("📊 Comprehensive Audit System")
        self.root.geometry("1200x800")

        # Initialize components
        print("[DEBUG] Initializing backup manager...")
        self.config = ConfigManager()
        self.backup_manager = iCloudBackupManager(self.config.config)
        print("[DEBUG] Backup manager initialized")
        self.audit_engine = None
        self.report_generator = None
        self.progress_queue = queue.Queue()
        self.result_comboboxes = {}  # Store comboboxes per row
        self.result_apply_buttons = {}  # Store apply buttons per row

        # Create main notebook for tabs
        print("[DEBUG] Creating notebook and tabs...")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        # Create tabs
        self.audit_tab = ttk.Frame(self.notebook)
        self.backup_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.audit_tab, text='🔍 Audit')
        self.notebook.add(self.backup_tab, text='💾 Backup')
        self.notebook.add(self.settings_tab, text='⚙️ Settings')
        self.notebook.add(self.results_tab, text='Results')

        # Setup each tab
        self.setup_audit_tab()
        self.setup_backup_tab()
        self.setup_settings_tab()
        self.setup_results_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress queue for thread communication
        self.root.after(100, self.check_progress)
        print("[DEBUG] AuditUI.__init__ finished")

    def setup_audit_tab(self):
        """Setup the audit tab"""

        # Target directory selection
        dir_frame = ttk.LabelFrame(self.audit_tab, text="Target Directory")
        dir_frame.pack(fill='x', padx=5, pady=5)

        self.target_var = tk.StringVar(value="/Users/georgegiailoglou/Library/Mobile Documents/com~apple~CloudDocs/AdamsGames/1. Εγγραφα εταιριας")
        ttk.Entry(dir_frame, textvariable=self.target_var, width=50).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_target).pack(side=tk.LEFT, padx=5)

        # Output file selection
        output_frame = ttk.LabelFrame(self.audit_tab, text="Output File")
        output_frame.pack(fill='x', padx=5, pady=5)

        self.output_var = tk.StringVar(value="comprehensive_audit_report.xlsx")
        ttk.Entry(output_frame, textvariable=self.output_var, width=50).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=5)

        # Options frame
        options_frame = ttk.LabelFrame(self.audit_tab, text="Options")
        options_frame.pack(fill='x', padx=5, pady=5)

        self.use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use AI Features", variable=self.use_ai_var).pack(side=tk.LEFT, padx=5)

        self.zero_change_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Zero-Change Mode", variable=self.zero_change_var).pack(side=tk.LEFT, padx=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(self.audit_tab, text="Progress")
        progress_frame.pack(fill='x', padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(pady=5)

        # Action buttons
        button_frame = ttk.Frame(self.audit_tab)
        button_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(button_frame, text="Start Audit", command=self.start_audit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_audit).pack(side=tk.LEFT, padx=5)

    def setup_backup_tab(self):
        """Setup the backup tab"""

        # Backup source selection
        source_frame = ttk.LabelFrame(self.backup_tab, text="Backup Source")
        source_frame.pack(fill='x', padx=5, pady=5)

        self.backup_source_var = tk.StringVar(value="/Users/georgegiailoglou/Library/Mobile Documents/com~apple~CloudDocs/AdamsGames/1. Εγγραφα εταιριας")
        ttk.Entry(source_frame, textvariable=self.backup_source_var, width=50).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(source_frame, text="Browse", command=self.browse_backup_source).pack(side=tk.LEFT, padx=5)

        # iCloud folders list
        icloud_frame = ttk.LabelFrame(self.backup_tab, text="iCloud Folders")
        icloud_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.icloud_listbox = tk.Listbox(icloud_frame, height=10)
        self.icloud_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        self.icloud_listbox.bind('<<ListboxSelect>>', self.on_icloud_select)

        ttk.Button(icloud_frame, text="Refresh iCloud Folders", command=self.refresh_icloud_folders).pack(pady=5)

        # Backup options
        options_frame = ttk.LabelFrame(self.backup_tab, text="Backup Options")
        options_frame.pack(fill='x', padx=5, pady=5)

        self.backup_name_var = tk.StringVar(value=self.get_next_backup_name())
        ttk.Label(options_frame, text="Backup Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(options_frame, textvariable=self.backup_name_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(options_frame, text="Generate Name", command=lambda: self.backup_name_var.set(self.get_next_backup_name())).pack(side=tk.LEFT, padx=5)

        ttk.Button(options_frame, text="Λήψη μη τοπικών αρχείων", command=self.download_missing_files_ui).pack(side=tk.LEFT, padx=5)

        # Backup progress
        progress_frame = ttk.LabelFrame(self.backup_tab, text="Backup Progress")
        progress_frame.pack(fill='x', padx=5, pady=5)

        self.backup_progress_var = tk.DoubleVar()
        self.backup_progress_bar = ttk.Progressbar(progress_frame, variable=self.backup_progress_var, maximum=100)
        self.backup_progress_bar.pack(fill='x', padx=5, pady=5)

        self.backup_progress_label = ttk.Label(progress_frame, text="")
        self.backup_progress_label.pack(pady=5)

        # Backup actions
        action_frame = ttk.Frame(self.backup_tab)
        action_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(action_frame, text="Create Backup", command=self.create_backup).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="List Backups", command=self.list_backups).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Restore Backup", command=self.restore_backup).pack(side=tk.LEFT, padx=5)

        # Backup list
        list_frame = ttk.LabelFrame(self.backup_tab, text="Available Backups")
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.backup_listbox = tk.Listbox(list_frame, height=10)
        self.backup_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        self.backup_listbox.bind('<<ListboxSelect>>', self.on_backup_select)

        # Refresh backup list
        self.refresh_backup_list()

        # Restore label
        restore_label = ttk.Label(list_frame, text="Για restore από terminal: unzip <backup.zip> -d <φάκελος>", foreground="gray")
        restore_label.pack(anchor=tk.W, pady=(10,0))

    def setup_settings_tab(self):
        """Setup the settings tab"""

        # AI Settings
        ai_frame = ttk.LabelFrame(self.settings_tab, text="AI Settings")
        ai_frame.pack(fill='x', padx=5, pady=5)

        self.similarity_var = tk.DoubleVar(value=0.8)
        ttk.Label(ai_frame, text="Similarity Threshold:").pack(side=tk.LEFT, padx=5)
        ttk.Scale(ai_frame, from_=0.0, to=1.0, variable=self.similarity_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, padx=5, fill='x', expand=True)

        # Backup Settings
        backup_frame = ttk.LabelFrame(self.settings_tab, text="Backup Settings")
        backup_frame.pack(fill='x', padx=5, pady=5)

        self.backup_location_var = tk.StringVar(value=str(Path.home() / 'AuditBackups'))
        ttk.Label(backup_frame, text="Backup Location:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(backup_frame, textvariable=self.backup_location_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(backup_frame, text="Browse", command=self.browse_backup_location).pack(side=tk.LEFT, padx=5)

        # Save settings button
        ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings).pack(pady=10)

    def setup_results_tab(self):
        # Create Treeview for results
        self.results_tree = ttk.Treeview(self.results_tab, columns=('Filename', 'Document Type', 'Suggested Action', 'Explanation'), show='headings')
        self.results_tree.heading('Filename', text='Filename')
        self.results_tree.heading('Document Type', text='Document Type')
        self.results_tree.heading('Suggested Action', text='Suggested Action')
        self.results_tree.heading('Explanation', text='Explanation')

        # Configure column widths
        self.results_tree.column('Filename', width=200)
        self.results_tree.column('Document Type', width=150)
        self.results_tree.column('Suggested Action', width=150)
        self.results_tree.column('Explanation', width=400)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Pack components
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.results_tree.bind('<ButtonRelease-1>', self.on_result_select)

        # Add buttons frame
        buttons_frame = ttk.Frame(self.results_tab)
        buttons_frame.pack(side='bottom', fill='x', padx=5, pady=5)

        # Add buttons
        ttk.Button(buttons_frame, text="Apply All Suggestions", command=self.apply_all_suggestions).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Export Results", command=self.export_results).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Clear Results", command=self.clear_results).pack(side='left', padx=5)

    def browse_target(self):
        """Browse for target directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.target_var.set(directory)

    def browse_output(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.output_var.set(filename)

    def browse_backup_source(self):
        """Browse for backup source directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.backup_source_var.set(directory)

    def browse_backup_location(self):
        """Browse for backup location"""
        directory = filedialog.askdirectory()
        if directory:
            self.backup_location_var.set(directory)

    def refresh_icloud_folders(self):
        """Refresh the list of iCloud folders"""
        self.icloud_listbox.delete(0, tk.END)

        icloud_base = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs'
        if not icloud_base.exists():
            self.icloud_listbox.insert(tk.END, "No iCloud Drive found")
            return

        # Add iCloud folders
        for item in icloud_base.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                    self.icloud_listbox.insert(tk.END, f"{item.name} ({file_count} files)")
                except:
                    self.icloud_listbox.insert(tk.END, item.name)

    def on_icloud_select(self, event):
        """Handle iCloud folder selection"""
        selection = self.icloud_listbox.curselection()
        if selection:
            folder_name = self.icloud_listbox.get(selection[0]).split(' (')[0]
            icloud_path = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / folder_name
            self.backup_source_var.set(str(icloud_path))

    def refresh_backup_list(self):
        """Refresh the list of available backups"""
        self.backup_listbox.delete(0, tk.END)

        backups = self.backup_manager.list_backups()
        for backup in backups:
            self.backup_listbox.insert(tk.END, f"{backup['name']} ({backup['size_mb']:.1f} MB) - {backup['created'].strftime('%Y-%m-%d %H:%M')}")

    def on_backup_select(self, event):
        """Handle backup selection"""
        selection = self.backup_listbox.curselection()
        if selection:
            backup_name = self.backup_listbox.get(selection[0]).split(' (')[0]
            self.backup_name_var.set(backup_name)

    def create_backup(self):
        """Δημιουργεί backup του επιλεγμένου φακέλου."""
        backup_location = self.backup_source_var.get()
        if not backup_location:
            tk.messagebox.showerror("Σφάλμα", "Παρακαλώ επιλέξτε φάκελο για backup")
            return

        # Δημιουργία progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Πρόοδος Backup")
        progress_window.geometry("420x260")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.focus_force()  # Κάνει το παράθυρο πάντα ενεργό
        progress_window.lift()  # Φέρνει το παράθυρο μπροστά
        progress_window.attributes('-topmost', True)  # Κάνει το παράθυρο πάντα πάνω από άλλα

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)

        # Status label
        status_label = ttk.Label(progress_window, text="Προετοιμασία...")
        status_label.pack(pady=5)

        # Percentage label
        percent_label = ttk.Label(progress_window, text="0%")
        percent_label.pack(pady=5)

        # Live stats
        stats_frame = ttk.LabelFrame(progress_window, text="Στατιστικά Backup (Live)")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        files_label = ttk.Label(stats_frame, text="Αρχεία: ...")
        files_label.pack(anchor=tk.W)
        folders_label = ttk.Label(stats_frame, text="Φάκελοι: ...")
        folders_label.pack(anchor=tk.W)
        size_label = ttk.Label(stats_frame, text="Συνολικό μέγεθος: ... MB")
        size_label.pack(anchor=tk.W)
        zip_label = ttk.Label(stats_frame, text="Backup ZIP: ... MB")
        zip_label.pack(anchor=tk.W)

        # Για να ενημερώνουμε τα στατιστικά
        live_stats = {'files': 0, 'folders': 0, 'size': 0, 'zip': 0}

        def update_stats():
            files_label.config(text=f"Αρχεία: {live_stats['files']}")
            folders_label.config(text=f"Φάκελοι: {live_stats['folders']}")
            size_label.config(text=f"Συνολικό μέγεθος: {live_stats['size']:.2f} MB")
            zip_label.config(text=f"Backup ZIP: {live_stats['zip']:.2f} MB")
            progress_window.update_idletasks()

        def update_progress(message, progress):
            status_label.config(text=message)
            progress_var.set(progress)
            percent_label.config(text=f"{int(progress)}%")
            update_stats()

        try:
            # Υπολογισμός αρχείων/φακέλων/μεγέθους πριν το backup
            import os
            file_list, metadata_dict = self.backup_manager.get_files_and_metadata(backup_location)
            folder_set = set(os.path.dirname(f) for f in file_list)
            total_size = sum(os.path.getsize(f) for f in file_list if os.path.exists(f))
            live_stats['files'] = len(file_list)
            live_stats['folders'] = len(folder_set)
            live_stats['size'] = total_size / (1024 * 1024)
            update_stats()

            # Εκτέλεση backup
            summary = self.backup_manager.create_backup(
                backup_location,
                progress_callback=update_progress
            )
            # Ενημέρωση τελικού zip size
            live_stats['zip'] = summary['backup_zip_size_mb']
            update_stats()

            progress_window.attributes('-topmost', False)  # Αφαιρεί το topmost πριν κλείσει
            progress_window.destroy()
            msg = (
                f"Backup ολοκληρώθηκε!\n\n"
                f"Αρχεία: {summary['num_files']}\n"
                f"Φάκελοι: {summary['num_folders']}\n"
                f"Συνολικό μέγεθος: {summary['total_size_mb']} MB\n"
                f"Backup ZIP: {summary['backup_zip_size_mb']} MB\n"
                f"Backup path: {summary['backup_path']}\n"
                f"\nΑναλυτικό Report (CSV): {summary['csv_report']}\n"
                f"Αναλυτικό Report (TXT): {summary['txt_report']}\n"
            )
            if summary['failed_downloads']:
                msg += f"\n⚠️ Απέτυχαν να κατέβουν: {len(summary['failed_downloads'])} αρχεία"
            # Show summary and button to open report
            def open_csv():
                webbrowser.open(f"file://{summary['csv_report']}")
            def open_txt():
                webbrowser.open(f"file://{summary['txt_report']}")
            def open_folder():
                import subprocess
                subprocess.run(["open", "-R", summary['backup_path']])
            report_win = tk.Toplevel(self.root)
            report_win.title("Backup Summary & Report")
            report_win.geometry("600x360")
            tk.Label(report_win, text=msg, justify=tk.LEFT, anchor=tk.W, font=("Arial", 11)).pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            btn_frame = ttk.Frame(report_win)
            btn_frame.pack(pady=5)
            ttk.Button(btn_frame, text="Άνοιγμα CSV Report", command=open_csv).pack(side=tk.LEFT, padx=10)
            ttk.Button(btn_frame, text="Άνοιγμα TXT Report", command=open_txt).pack(side=tk.LEFT, padx=10)
            ttk.Button(btn_frame, text="Εμφάνιση Backup στο Finder", command=open_folder).pack(side=tk.LEFT, padx=10)
            self.refresh_backup_list()
        except Exception as e:
            progress_window.attributes('-topmost', False)  # Αφαιρεί το topmost πριν κλείσει
            progress_window.destroy()
            tk.messagebox.showerror("Σφάλμα", f"Σφάλμα κατά τη δημιουργία backup: {str(e)}")

    def list_backups(self):
        """List available backups"""
        self.refresh_backup_list()

    def restore_backup(self):
        """Restore selected backup"""
        selection = self.backup_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a backup to restore")
            return

        backup_name = self.backup_listbox.get(selection[0]).split(' (')[0]
        restore_dir = filedialog.askdirectory(title="Select restore location")

        if not restore_dir:
            return

        # Start restore in thread
        def restore_thread():
            try:
                def progress_callback(current, total, message):
                    self.progress_queue.put(('restore', current, total, message))

                success, result = self.backup_manager.restore_backup(
                    Path(self.backup_manager.backup_root / f"{backup_name}.tar.gz"),
                    Path(restore_dir),
                    progress_callback
                )

                if success:
                    self.progress_queue.put(('restore_complete', True, result))
                else:
                    self.progress_queue.put(('restore_complete', False, result))

            except Exception as e:
                self.progress_queue.put(('restore_error', str(e)))

        # Start thread
        self.backup_progress_var.set(0)
        self.backup_progress_label.config(text="Starting restore...")
        threading.Thread(target=restore_thread, daemon=True).start()

    def start_audit(self):
        """Start the audit process"""
        target_dir = self.target_var.get()
        if not target_dir:
            messagebox.showerror("Error", "Please select a target directory")
            return

        output_file = os.path.join(target_dir, f"audit_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        # Start audit in a separate thread
        def audit_thread():
            try:
                self.audit_engine = ComprehensiveAuditEngine(self.config)
                audit_results = self.audit_engine.run_comprehensive_audit(
                    target_dir,
                    progress_callback=lambda cur, tot, msg: self.progress_queue.put(('audit', cur, tot, msg))
                )
                if not self.audit_engine.cancelled:
                    self.report_generator = ExcelReportGenerator()
                    report_file = self.report_generator.generate_comprehensive_report(
                        audit_results, output_file
                    )
                    self.progress_queue.put(('audit_complete', True, report_file))

                    # Show results and apply suggestions automatically
                    self.show_audit_results(audit_results)
                    self.apply_all_suggestions()
                else:
                    self.progress_queue.put(('audit_complete', False, "Audit cancelled"))
            except Exception as e:
                self.progress_queue.put(('audit_error', str(e)))

        # Start the thread
        threading.Thread(target=audit_thread, daemon=True).start()

    def cancel_audit(self):
        """Cancel running audit"""
        if self.audit_engine:
            self.audit_engine.cancelled = True
            self.status_var.set("Cancelling audit...")

    def save_settings(self):
        """Save settings to file"""
        settings = {
            'similarity_threshold': self.similarity_var.get(),
            'backup_location': self.backup_location_var.get()
        }

        try:
            with open('audit_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def check_progress(self):
        """Check progress queue for updates"""
        try:
            while True:
                msg_type, *args = self.progress_queue.get_nowait()

                if msg_type == 'audit':
                    current, total, message = args
                    progress = (current / total) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=message)
                    self.status_var.set(f"Audit: {message}")

                elif msg_type == 'audit_complete':
                    success, result = args
                    if success:
                        messagebox.showinfo("Success", f"Audit completed successfully!\nReport saved: {result}")
                    else:
                        messagebox.showinfo("Cancelled", "Audit was cancelled")
                    self.progress_var.set(0)
                    self.progress_label.config(text="")
                    self.status_var.set("Ready")

                elif msg_type == 'audit_error':
                    error = args[0]
                    messagebox.showerror("Error", f"Audit failed: {error}")
                    self.progress_var.set(0)
                    self.progress_label.config(text="")
                    self.status_var.set("Error")

                elif msg_type == 'backup':
                    current, total, message = args
                    progress = (current / total) * 100
                    self.backup_progress_var.set(progress)
                    self.backup_progress_label.config(text=message)
                    self.status_var.set(f"Backup: {message}")

                elif msg_type == 'backup_complete':
                    success, backup_path, backup_info = args
                    if success:
                        messagebox.showinfo("Success",
                            f"Backup completed successfully!\n"
                            f"File: {backup_path}\n"
                            f"Size: {backup_info.get('backup_size_mb', 0):.1f} MB"
                        )
                        self.refresh_backup_list()
                    else:
                        messagebox.showerror("Error", f"Backup failed: {backup_path}")
                    self.backup_progress_var.set(0)
                    self.backup_progress_label.config(text="")
                    self.status_var.set("Ready")

                elif msg_type == 'backup_error':
                    error = args[0]
                    messagebox.showerror("Error", f"Backup failed: {error}")
                    self.backup_progress_var.set(0)
                    self.backup_progress_label.config(text="")
                    self.status_var.set("Error")

                elif msg_type == 'restore':
                    current, total, message = args
                    progress = (current / total) * 100
                    self.backup_progress_var.set(progress)
                    self.backup_progress_label.config(text=message)
                    self.status_var.set(f"Restore: {message}")

                elif msg_type == 'restore_complete':
                    success, result = args
                    if success:
                        messagebox.showinfo("Success", f"Restore completed successfully!\nRestored to: {result}")
                    else:
                        messagebox.showerror("Error", f"Restore failed: {result}")
                    self.backup_progress_var.set(0)
                    self.backup_progress_label.config(text="")
                    self.status_var.set("Ready")

                elif msg_type == 'restore_error':
                    error = args[0]
                    messagebox.showerror("Error", f"Restore failed: {error}")
                    self.backup_progress_var.set(0)
                    self.backup_progress_label.config(text="")
                    self.status_var.set("Error")

        except queue.Empty:
            pass

        self.root.after(100, self.check_progress)

    def get_next_backup_name(self) -> str:
        """Generate next backup name with date and sequential number"""
        today = datetime.datetime.now().strftime("%d_%m_%Y")

        # Get existing backups for today
        existing_backups = []
        for backup in self.backup_manager.list_backups():
            if backup['name'].startswith(f"backup_{today}"):
                try:
                    # Extract number from name like "backup_dd_mm_yyyy-1"
                    num = int(backup['name'].split('-')[-1])
                    existing_backups.append(num)
                except:
                    continue

        # Get next number
        next_num = 1
        if existing_backups:
            next_num = max(existing_backups) + 1

        return f"backup_{today}-{next_num}"

    def download_missing_files_ui(self):
        """Κατεβάζει αυτόματα όλα τα μη τοπικά αρχεία iCloud για τον επιλεγμένο φάκελο, με progress και αναλυτικό αποτέλεσμα."""
        backup_location = self.backup_source_var.get()
        if not backup_location:
            tk.messagebox.showerror("Σφάλμα", "Παρακαλώ επιλέξτε φάκελο")
            return

        # Δημιουργία progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Λήψη αρχείων από iCloud")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.focus_force()
        progress_window.lift()
        progress_window.attributes('-topmost', True)

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)

        # Status label
        status_label = ttk.Label(progress_window, text="Έλεγχος αρχείων...")
        status_label.pack(pady=5)

        # Percentage label
        percent_label = ttk.Label(progress_window, text="0%")
        percent_label.pack(pady=5)

        def update_progress(message, progress):
            status_label.config(text=message)
            progress_var.set(progress)
            percent_label.config(text=f"{int(progress)}%")
            progress_window.update_idletasks()

        try:
            file_list, metadata_dict = self.backup_manager.get_files_and_metadata(backup_location)
            failed = self.backup_manager.download_missing_files(file_list, metadata_dict, threads=8, progress_callback=update_progress)
            progress_window.attributes('-topmost', False)
            progress_window.destroy()
            if failed:
                msg = f"⚠️ Τα παρακάτω {len(failed)} αρχεία ΔΕΝ κατέβηκαν:\n\n" + '\n'.join(failed[:10])
                if len(failed) > 10:
                    msg += f"\n... ({len(failed)-10} ακόμη)"
                tk.messagebox.showwarning("Αρχεία που δεν κατέβηκαν", msg)
            else:
                tk.messagebox.showinfo("Επιτυχία", "Όλα τα αρχεία είναι τοπικά διαθέσιμα!")
        except Exception as e:
            progress_window.attributes('-topmost', False)
            progress_window.destroy()
            tk.messagebox.showerror("Σφάλμα", f"Σφάλμα κατά τη λήψη αρχείων: {str(e)}")

    def log_feedback(self, filename, doc_type, action, explanation):
        with open('feedback_log.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, doc_type, action, explanation])

    def apply_trainer_suggestions(self, files):
        trainer = FeedbackTrainer()
        results = []
        for file_info in files:
            fname = file_info.get('filename', '')
            doc_type = file_info.get('document_type', '')
            action, explanation = trainer.suggest_action(fname, doc_type)
            self.log_feedback(fname, doc_type, action, '; '.join(explanation))
            file_info['suggested_action'] = action
            file_info['explanation'] = '; '.join(explanation)
            results.append(file_info)
        return results

    def show_audit_results(self, audit_results):
        self.clear_results()
        self.result_comboboxes = {}
        self.result_apply_buttons = {}
        files = audit_results.get('files_analyzed', [])
        files_with_suggestions = self.apply_trainer_suggestions(files)
        for idx, file_info in enumerate(files_with_suggestions):
            row_id = self.results_tree.insert('', 'end', values=(
                file_info.get('filename', ''),
                file_info.get('document_type', ''),
                file_info.get('suggested_action', ''),
                file_info.get('explanation', '')
            ))
            # Prepare folder options
            folder_options = self.get_folder_options_for_file(file_info)
            # Place Combobox and Apply button
            self.root.after(100, lambda row_id=row_id, options=folder_options, file_info=file_info: self.add_combobox_and_button(row_id, options, file_info))
        self.notebook.select(self.results_tab)

    def get_folder_options_for_file(self, file_info):
        """Return all relevant folder options for a file (current, suggested, Invoice, Container, HBL, etc)"""
        options = set()
        current_folder = os.path.dirname(file_info.get('full_path', ''))
        options.add(current_folder)
        # Add suggested folder if exists
        suggestion = file_info.get('suggested_action', '')
        explanation = file_info.get('explanation', '')
        if 'Move to' in explanation:
            parts = explanation.split('Move to')
            if len(parts) > 1:
                folder_name = parts[1].split('folder')[0].strip()
                parent = os.path.dirname(current_folder)
                suggested_folder = os.path.join(parent, folder_name)
                options.add(suggested_folder)
        # Add Invoice folder if relevant
        if file_info.get('document_type', '').lower() == 'invoice':
            parent = os.path.dirname(current_folder)
            invoice_folder = os.path.join(parent, 'Invoice')
            options.add(invoice_folder)
        # Add container/HBL folders if found in explanation
        for word in explanation.split():
            if word.startswith('Container') or word.startswith('HBL'):
                parent = os.path.dirname(current_folder)
                folder = os.path.join(parent, word)
                options.add(folder)
        return list(options)

    def add_combobox_and_button(self, row_id, options, file_info):
        bbox = self.results_tree.bbox(row_id, column=2)
        if not bbox:
            self.root.after(100, lambda: self.add_combobox_and_button(row_id, options, file_info))
            return
        x, y, width, height = bbox
        combobox = ttk.Combobox(self.results_tree, values=options, width=30)
        combobox.set(options[0] if options else '')
        combobox.place(x=x, y=y, width=width, height=height)
        self.result_comboboxes[row_id] = combobox
        # Add Apply button
        btn = ttk.Button(self.results_tree, text="Εφαρμογή", command=lambda rid=row_id, fi=file_info: self.apply_single_suggestion(rid, fi))
        btn.place(x=x+width+5, y=y, width=80, height=height)
        self.result_apply_buttons[row_id] = btn

    def on_result_select(self, event):
        # Reposition comboboxes/buttons if needed (refresh/repaint)
        for row_id, combobox in self.result_comboboxes.items():
            bbox = self.results_tree.bbox(row_id, column=2)
            if bbox:
                x, y, width, height = bbox
                combobox.place(x=x, y=y, width=width, height=height)
                btn = self.result_apply_buttons.get(row_id)
                if btn:
                    btn.place(x=x+width+5, y=y, width=80, height=height)
        self.results_tree.update_idletasks()

    def apply_single_suggestion(self, row_id, file_info):
        combobox = self.result_comboboxes.get(row_id)
        if not combobox:
            return
        target_folder = combobox.get()
        filename = file_info.get('filename', '')
        source_path = file_info.get('full_path', '')
        dest_path = os.path.join(target_folder, filename)
        try:
            os.makedirs(target_folder, exist_ok=True)
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(target_folder, f"{base}_{timestamp}{ext}")
            os.rename(source_path, dest_path)
            messagebox.showinfo("Επιτυχία", f"Το αρχείο μετακινήθηκε στο {target_folder}")
            self.results_tree.delete(row_id)
            if row_id in self.result_comboboxes:
                self.result_comboboxes[row_id].destroy()
                del self.result_comboboxes[row_id]
            if row_id in self.result_apply_buttons:
                self.result_apply_buttons[row_id].destroy()
                del self.result_apply_buttons[row_id]
        except Exception as e:
            messagebox.showerror("Σφάλμα", f"Αποτυχία μετακίνησης: {str(e)}")

    def apply_all_suggestions(self):
        """Apply all actions based on the current selections in the comboboxes"""
        to_remove = []
        for row_id, combobox in self.result_comboboxes.items():
            target_folder = combobox.get()
            file_info = None
            # Find file_info by row_id (from the button dict)
            for rid, btn in self.result_apply_buttons.items():
                if rid == row_id:
                    file_info = btn['command'].__closure__[1].cell_contents if hasattr(btn, 'command') else None
                    break
            if not file_info:
                continue
            filename = file_info.get('filename', '')
            source_path = file_info.get('full_path', '')
            dest_path = os.path.join(target_folder, filename)
            try:
                os.makedirs(target_folder, exist_ok=True)
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest_path = os.path.join(target_folder, f"{base}_{timestamp}{ext}")
                os.rename(source_path, dest_path)
                to_remove.append(row_id)
            except Exception as e:
                messagebox.showerror("Σφάλμα", f"Αποτυχία μετακίνησης του {filename}: {str(e)}")
        # Remove rows and destroy widgets
        for row_id in to_remove:
            self.results_tree.delete(row_id)
            if row_id in self.result_comboboxes:
                self.result_comboboxes[row_id].destroy()
                del self.result_comboboxes[row_id]
            if row_id in self.result_apply_buttons:
                self.result_apply_buttons[row_id].destroy()
                del self.result_apply_buttons[row_id]
        if to_remove:
            messagebox.showinfo("Επιτυχία", "Όλα τα επιλεγμένα αρχεία μετακινήθηκαν!")

    def export_results(self):
        """Export results to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Results"
        )
        if filename:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Document Type', 'Suggested Action', 'Explanation'])
                for item in self.results_tree.get_children():
                    writer.writerow(self.results_tree.item(item)['values'])

    def clear_results(self):
        """Clear all results from the treeview"""
        self.results_tree.delete(*self.results_tree.get_children())

def main():
    """Main function"""

    print("[DEBUG] main() started")
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return 1

    root = tk.Tk()
    print("[DEBUG] Tkinter root created")
    app = AuditUI(root)
    print("[DEBUG] AuditUI instance created")
    root.mainloop()
    print("[DEBUG] mainloop exited")
    return 0

if __name__ == '__main__':
    sys.exit(main())