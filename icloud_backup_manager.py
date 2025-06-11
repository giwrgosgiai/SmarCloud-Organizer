#!/usr/bin/env python3
"""
💾 iCloud-AWARE BACKUP MANAGER
🍎 Mac-optimized backup system with iCloud Drive support
🔄 Full restore capability with metadata preservation
"""

import os
import sys
import shutil
import subprocess
import json
import tarfile
import zipfile
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import xattr
import threading
from queue import Queue
import csv

class iCloudBackupManager:
    """Mac-optimized backup manager with iCloud Drive support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cancelled = False
        self.progress_callback = None
        self.download_queue = Queue()
        self.download_threads = []
        self.logger = logging.getLogger(__name__)
        self.backup_root = Path.home() / 'AuditBackups'
        self.backup_root.mkdir(exist_ok=True)

        # iCloud Drive path detection
        self.icloud_drive_path = self._detect_icloud_drive()

    def _detect_icloud_drive(self) -> Optional[Path]:
        """Detect iCloud Drive path on Mac"""
        possible_paths = [
            Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs',
            Path.home() / 'iCloud Drive (Archive)',
            Path.home() / 'iCloud Drive'
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"📱 Detected iCloud Drive: {path}")
                return path

        self.logger.warning("⚠️  iCloud Drive not detected")
        return None

    def is_icloud_optimized_file(self, file_path: Path) -> bool:
        """Check if file is iCloud optimized (cloud-only)"""
        try:
            # Check for .icloud placeholder files
            if file_path.name.startswith('.') and file_path.name.endswith('.icloud'):
                return True

            # Check file attributes using xattr
            result = subprocess.run([
                'xattr', '-l', str(file_path)
            ], capture_output=True, text=True, timeout=5)

            # Look for iCloud attributes
            if 'com.apple.ubiquity' in result.stdout:
                return True

            return False

        except Exception as e:
            self.logger.debug(f"Could not check iCloud status for {file_path}: {e}")
            return False

    def download_icloud_file(self, file_path: Path) -> bool:
        """Force download of iCloud optimized file"""
        try:
            self.logger.info(f"📥 Downloading from iCloud: {file_path.name}")

            # Use brctl (if available) to download
            result = subprocess.run([
                'brctl', 'download', str(file_path)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                self.logger.info(f"✅ Downloaded: {file_path.name}")
                return True
            else:
                # Fallback: Try to read the file to trigger download
                try:
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Read first 1KB to trigger download
                    return True
                except:
                    pass

        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ Timeout downloading: {file_path.name}")
        except Exception as e:
            self.logger.warning(f"❌ Failed to download {file_path.name}: {e}")

        return False

    def scan_and_download_icloud_files(self, target_dir: Path, progress_callback=None) -> Dict:
        """Scan directory and download iCloud optimized files"""

        self.logger.info(f"🔍 Scanning for iCloud optimized files in: {target_dir}")

        icloud_files = []
        downloaded_files = []
        failed_files = []

        # Find all files
        all_files = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                all_files.append(Path(root) / file)

        total_files = len(all_files)

        for i, file_path in enumerate(all_files):
            if progress_callback:
                progress_callback(i + 1, total_files, f"Checking: {file_path.name}")

            if self.is_icloud_optimized_file(file_path):
                icloud_files.append(file_path)

                if self.download_icloud_file(file_path):
                    downloaded_files.append(file_path)
                else:
                    failed_files.append(file_path)

        return {
            'total_files': total_files,
            'icloud_files': len(icloud_files),
            'downloaded': len(downloaded_files),
            'failed': len(failed_files),
            'failed_files': failed_files
        }

    def create_backup_metadata(self, source_dir: Path, backup_info: Dict) -> Dict:
        """Create comprehensive backup metadata"""

        metadata = {
            'backup_timestamp': datetime.now().isoformat(),
            'source_directory': str(source_dir),
            'backup_type': 'comprehensive_audit_backup',
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'user': os.getenv('USER', 'unknown')
            },
            'icloud_info': {
                'icloud_drive_detected': self.icloud_drive_path is not None,
                'icloud_drive_path': str(self.icloud_drive_path) if self.icloud_drive_path else None,
                'files_downloaded': backup_info.get('downloaded', 0),
                'failed_downloads': backup_info.get('failed', 0)
            },
            'file_counts': {
                'total_files': backup_info.get('total_files', 0),
                'directories': 0,
                'size_bytes': 0
            },
            'checksums': {}
        }

        # Calculate directory stats
        total_size = 0
        dir_count = 0
        file_checksums = {}

        for root, dirs, files in os.walk(source_dir):
            dir_count += len(dirs)
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size

                    # Calculate checksum for smaller files
                    if file_size < 10 * 1024 * 1024:  # 10MB limit
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            relative_path = file_path.relative_to(source_dir)
                            file_checksums[str(relative_path)] = file_hash

                except Exception as e:
                    self.logger.warning(f"Could not process {file_path}: {e}")

        metadata['file_counts']['directories'] = dir_count
        metadata['file_counts']['size_bytes'] = total_size
        metadata['checksums'] = file_checksums

        return metadata

    def get_files_and_metadata(self, backup_location):
        """Επιστρέφει λίστα με paths και dictionary με metadata για κάθε αρχείο."""
        file_list = []
        metadata_dict = {}

        for root, _, files in os.walk(backup_location):
            for file in files:
                if file.endswith('.DS_Store'):
                    continue
                full_path = os.path.join(root, file)
                try:
                    stat = os.stat(full_path)
                    metadata_dict[full_path] = {
                        'size': stat.st_size,
                        'mtime': int(stat.st_mtime)
                    }
                    file_list.append(full_path)
                except Exception as e:
                    print(f"Error getting metadata for {full_path}: {e}")

        return file_list, metadata_dict

    def create_backup(self, source_dir: str, output_file: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Create a backup of the source directory, handling iCloud files.

        Args:
            source_dir: Directory to backup
            output_file: Path to save the backup ZIP
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with backup results
        """
        self.progress_callback = progress_callback
        self.cancelled = False

        results = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': source_dir,
            'output_file': output_file,
            'files_backed_up': [],
            'files_failed': [],
            'total_size': 0,
            'zip_size': 0
        }

        # Get all files
        all_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        total_files = len(all_files)

        # Create temporary directory for downloaded files
        temp_dir = os.path.join(os.path.dirname(output_file), f"temp_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Start download threads
            num_threads = min(4, total_files)  # Use up to 4 threads
            for _ in range(num_threads):
                thread = threading.Thread(target=self._download_worker, args=(temp_dir,))
                thread.daemon = True
                thread.start()
                self.download_threads.append(thread)

            # Process files
            for idx, file_path in enumerate(all_files, 1):
                if self.cancelled:
                    break

                if self.progress_callback:
                    self.progress_callback(idx, total_files, f"Processing {os.path.basename(file_path)}")

                try:
                    # Check if file is in iCloud
                    if self._is_icloud_file(file_path):
                        # Add to download queue
                        self.download_queue.put(file_path)
                    else:
                        # File is local, copy to temp directory
                        rel_path = os.path.relpath(file_path, source_dir)
                        temp_path = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        shutil.copy2(file_path, temp_path)
                        results['files_backed_up'].append(rel_path)
                        results['total_size'] += os.path.getsize(file_path)

                except Exception as e:
                    results['files_failed'].append({
                        'file': file_path,
                        'error': str(e)
                    })

            # Wait for downloads to complete
            self.download_queue.join()

            # Create ZIP file
            if self.progress_callback:
                self.progress_callback(0, 1, "Creating backup ZIP...")

            with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)

            results['zip_size'] = os.path.getsize(output_file)

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {str(e)}")

        return results

    def _is_icloud_file(self, file_path: str) -> bool:
        """Check if a file is in iCloud"""
        try:
            # Check for iCloud-specific attributes
            # This is a simplified check - you might need to adjust based on your system
            return os.path.getxattr(file_path, 'com.apple.metadata:kMDItemIsInCloud') == b'1'
        except:
            return False

    def _download_worker(self, temp_dir: str):
        """Worker thread for downloading iCloud files"""
        while not self.cancelled:
            try:
                file_path = self.download_queue.get(timeout=1)

                try:
                    # Create relative path in temp directory
                    rel_path = os.path.relpath(file_path, self.config.get('source_directory', ''))
                    temp_path = os.path.join(temp_dir, rel_path)

                    # Create directory if needed
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

                    # Copy file (this will trigger download if needed)
                    shutil.copy2(file_path, temp_path)

                except Exception as e:
                    print(f"Error downloading {file_path}: {str(e)}")

                finally:
                    self.download_queue.task_done()

            except Queue.Empty:
                break

    def cancel(self):
        """Cancel the current backup"""
        self.cancelled = True
        # Clear download queue
        while not self.download_queue.empty():
            try:
                self.download_queue.get_nowait()
                self.download_queue.task_done()
            except Queue.Empty:
                break

    def list_backups(self) -> List[Dict]:
        """List all available backups"""

        backups = []

        for backup_file in self.backup_root.glob("*.tar.gz"):
            backup_name = backup_file.stem
            metadata_file = self.backup_root / f"{backup_name}_metadata.json"

            backup_info = {
                'name': backup_name,
                'file': str(backup_file),
                'size_mb': backup_file.stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(backup_file.stat().st_ctime),
                'metadata_available': metadata_file.exists()
            }

            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        backup_info['source_directory'] = metadata.get('source_directory')
                        backup_info['icloud_files'] = metadata.get('icloud_info', {}).get('files_downloaded', 0)
                        backup_info['total_files'] = metadata.get('file_counts', {}).get('total_files', 0)
                except Exception as e:
                    self.logger.warning(f"Could not load metadata for {backup_name}: {e}")

            backups.append(backup_info)

        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)

        return backups

    def restore_backup(self, backup_file: Path, restore_dir: Path,
                      progress_callback=None) -> Tuple[bool, str]:
        """Restore backup to specified directory"""

        if not backup_file.exists():
            return False, f"Backup file not found: {backup_file}"

        if restore_dir.exists() and any(restore_dir.iterdir()):
            return False, f"Restore directory is not empty: {restore_dir}"

        try:
            restore_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"🔄 Restoring backup: {backup_file} → {restore_dir}")

            if progress_callback:
                progress_callback(0, 100, "Extracting archive...")

            with tarfile.open(backup_file, 'r:gz') as tar:
                # Get list of members for progress tracking
                members = tar.getmembers()
                total_members = len(members)

                for i, member in enumerate(members):
                    if progress_callback:
                        progress = int((i / total_members) * 100)
                        progress_callback(progress, 100, f"Extracting: {member.name}")

                    tar.extract(member, restore_dir)

            if progress_callback:
                progress_callback(100, 100, "Restore completed!")

            self.logger.info(f"✅ Restore completed: {restore_dir}")
            return True, str(restore_dir)

        except Exception as e:
            self.logger.error(f"❌ Restore failed: {e}")
            return False, str(e)

    def verify_backup(self, backup_file: Path) -> Dict:
        """Verify backup integrity"""

        if not backup_file.exists():
            return {'valid': False, 'error': 'Backup file not found'}

        try:
            # Check if tar file is valid
            with tarfile.open(backup_file, 'r:gz') as tar:
                # Quick integrity check
                tar.getmembers()

                # Get basic stats
                members = tar.getmembers()
                total_files = len([m for m in members if m.isfile()])
                total_dirs = len([m for m in members if m.isdir()])
                total_size = sum(m.size for m in members if m.isfile())

            # Check metadata file
            backup_name = backup_file.stem
            metadata_file = backup_file.parent / f"{backup_name}_metadata.json"

            metadata_valid = False
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        metadata_valid = True
                except:
                    pass

            return {
                'valid': True,
                'total_files': total_files,
                'total_directories': total_dirs,
                'total_size_bytes': total_size,
                'metadata_valid': metadata_valid,
                'backup_size_bytes': backup_file.stat().st_size
            }

        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def is_file_local(self, path):
        """Επιστρέφει True αν το αρχείο είναι τοπικά διαθέσιμο (όχι μόνο στο iCloud)."""
        try:
            attrs = xattr.xattr(path)
            return b'com.apple.metadata:com_apple_cloud_docs' not in attrs
        except Exception:
            return True  # Αν δεν υπάρχουν xattr, θεωρούμε ότι είναι τοπικό

    def has_file_changed(self, path, metadata):
        """Επιστρέφει True αν το αρχείο έχει αλλάξει σε σχέση με το metadata (μέγεθος, mtime)."""
        try:
            stat = os.stat(path)
            if stat.st_size != metadata.get('size'):
                return True
            if int(stat.st_mtime) != metadata.get('mtime'):
                return True
            return False
        except Exception:
            return True

    def download_file(self, path):
        """Ζητάει από το macOS να κατεβάσει το αρχείο από το iCloud (αν δεν είναι τοπικό)."""
        try:
            subprocess.run(['xattr', '-p', 'com.apple.metadata:com_apple_cloud_docs', path], check=True)
            subprocess.run(['brctl', 'download', path], check=False)
        except Exception:
            pass

    def download_missing_files(self, file_list, metadata_dict, threads=8, progress_callback=None):
        """Κατεβάζει όσα αρχεία δεν είναι τοπικά και έχουν αλλάξει, με multithreading. Επιστρέφει λίστα με όσα απέτυχαν."""
        failed = []
        to_download = []
        for path in file_list:
            if not self.is_file_local(path):
                meta = metadata_dict.get(path, {})
                if self.has_file_changed(path, meta):
                    to_download.append(path)
        total = len(to_download)
        if total == 0:
            if progress_callback:
                progress_callback("Όλα τα αρχεία είναι ήδη τοπικά!", 100)
            return []
        q = Queue()
        for p in to_download:
            q.put(p)
        lock = threading.Lock()
        progress = [0]
        def worker():
            while not q.empty():
                path = q.get()
                self.download_file(path)
                with lock:
                    progress[0] += 1
                    if progress_callback:
                        percent = int((progress[0]/total)*100)
                        progress_callback(f"Λήψη iCloud αρχείων... ({progress[0]}/{total})", percent)
                if not self.is_file_local(path):
                    with lock:
                        failed.append(path)
                q.task_done()
        threads_list = []
        for _ in range(min(threads, total)):
            t = threading.Thread(target=worker)
            t.start()
            threads_list.append(t)
        q.join()
        for t in threads_list:
            t.join()
        if progress_callback:
            progress_callback("Λήψη iCloud αρχείων ολοκληρώθηκε!", 100)
        return failed

def create_progress_callback(queue: Queue) -> Callable:
    """Create a progress callback function that puts updates in a queue"""
    def callback(current: int, total: int, message: str):
        queue.put(('backup', current, total, message))
    return callback