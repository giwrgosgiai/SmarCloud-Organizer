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

try:
    from advanced_audit_system import ComprehensiveAuditEngine
    from excel_report_generator import ExcelReportGenerator
    from icloud_backup_manager import iCloudBackupManager, create_progress_callback
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    MODULES_AVAILABLE = False

class AuditUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Comprehensive Audit System")
        self.root.geometry("1200x800")

        # Initialize components
        self.backup_manager = iCloudBackupManager()
        self.audit_engine = None
        self.report_generator = None

        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        # Create tabs
        self.audit_tab = ttk.Frame(self.notebook)
        self.backup_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.audit_tab, text='🔍 Audit')
        self.notebook.add(self.backup_tab, text='💾 Backup')
        self.notebook.add(self.settings_tab, text='⚙️ Settings')

        # Setup each tab
        self.setup_audit_tab()
        self.setup_backup_tab()
        self.setup_settings_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress queue for thread communication
        self.progress_queue = queue.Queue()
        self.root.after(100, self.check_progress)

    def setup_audit_tab(self):
        """Setup the audit tab"""

        # Target directory selection
        dir_frame = ttk.LabelFrame(self.audit_tab, text="Target Directory")
        dir_frame.pack(fill='x', padx=5, pady=5)

        self.target_var = tk.StringVar()
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
        progress_window.geometry("420x220")
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
            )
            if summary['failed_downloads']:
                msg += f"\n⚠️ Απέτυχαν να κατέβουν: {len(summary['failed_downloads'])} αρχεία"
            tk.messagebox.showinfo("Backup Summary", msg)
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
        """Start audit process"""
        target_dir = self.target_var.get()
        if not target_dir:
            messagebox.showerror("Error", "Please select a target directory")
            return

        output_file = self.output_var.get()
        if not output_file:
            messagebox.showerror("Error", "Please specify an output file")
            return

        # Create config
        config = {
            'enable_ai': self.use_ai_var.get(),
            'zero_change_mode': self.zero_change_var.get(),
            'similarity_threshold': self.similarity_var.get()
        }

        # Start audit in thread
        def audit_thread():
            try:
                self.audit_engine = ComprehensiveAuditEngine(config)

                def progress_callback(current, total, message):
                    self.progress_queue.put(('audit', current, total, message))

                audit_results = self.audit_engine.run_comprehensive_audit(
                    target_dir, progress_callback=progress_callback
                )

                if not self.audit_engine.cancelled:
                    self.report_generator = ExcelReportGenerator()
                    report_file = self.report_generator.generate_comprehensive_report(
                        audit_results, output_file
                    )
                    self.progress_queue.put(('audit_complete', True, report_file))
                else:
                    self.progress_queue.put(('audit_complete', False, "Audit cancelled"))

            except Exception as e:
                self.progress_queue.put(('audit_error', str(e)))

        # Start thread
        self.progress_var.set(0)
        self.progress_label.config(text="Starting audit...")
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

def main():
    """Main function"""

    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return 1

    root = tk.Tk()
    app = AuditUI(root)
    root.mainloop()
    return 0

if __name__ == '__main__':
    sys.exit(main())