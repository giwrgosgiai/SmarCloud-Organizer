#!/usr/bin/env python3
"""
💾 INTERACTIVE BACKUP UI
🎛️  Easy backup management for iCloud Drive folders
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import datetime

try:
    from icloud_backup_manager import iCloudBackupManager, create_progress_callback
    BACKUP_AVAILABLE = True
except ImportError:
    print("❌ Backup manager not available")
    BACKUP_AVAILABLE = False

def print_header():
    """Print the backup UI header"""
    print("\n" + "="*80)
    print("💾 iCLOUD BACKUP MANAGER")
    print("🍎 Mac-optimized backup system with iCloud Drive support")
    print("="*80)

def detect_icloud_folders() -> List[Path]:
    """Detect common iCloud Drive folders for business documents"""

    icloud_base = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs'

    if not icloud_base.exists():
        return []

    # Common business document folder patterns
    candidate_folders = []

    # Direct iCloud subfolders
    for item in icloud_base.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            candidate_folders.append(item)

    # Look for business-like folders
    business_keywords = [
        'business', 'company', 'work', 'documents', 'files',
        'customers', 'invoices', 'contracts', 'projects',
        'επιχειρηση', 'εταιρια', 'εργασια', 'αρχεια', 'πελατες'
    ]

    business_folders = []
    for folder in candidate_folders:
        folder_name_lower = folder.name.lower()
        if any(keyword in folder_name_lower for keyword in business_keywords):
            business_folders.append(folder)

    # Add all folders but prioritize business ones
    all_folders = business_folders + [f for f in candidate_folders if f not in business_folders]

    return all_folders[:10]  # Limit to top 10

def select_folder_interactive() -> Optional[Path]:
    """Interactive folder selection"""

    print("\n📂 FOLDER SELECTION:")
    print("-" * 40)

    # Method 1: Detect iCloud folders
    icloud_folders = detect_icloud_folders()

    if icloud_folders:
        print("\n🔍 Detected iCloud Drive folders:")
        for i, folder in enumerate(icloud_folders, 1):
            # Calculate folder size
            try:
                file_count = sum(1 for _ in folder.rglob('*') if _.is_file())
                print(f"   {i:2d}. {folder.name} ({file_count} files)")
            except:
                print(f"   {i:2d}. {folder.name}")

        print(f"   {len(icloud_folders)+1:2d}. 🔍 Browse for custom folder")
        print(f"   {len(icloud_folders)+2:2d}. ❌ Cancel")

        while True:
            try:
                choice = input(f"\nSelect folder (1-{len(icloud_folders)+2}): ").strip()

                if not choice:
                    continue

                choice_num = int(choice)

                if 1 <= choice_num <= len(icloud_folders):
                    selected_folder = icloud_folders[choice_num - 1]
                    print(f"✅ Selected: {selected_folder}")
                    return selected_folder

                elif choice_num == len(icloud_folders) + 1:
                    # Browse for custom folder
                    custom_path = input("\n📁 Enter full path to folder: ").strip()
                    if custom_path:
                        custom_folder = Path(custom_path)
                        if custom_folder.exists() and custom_folder.is_dir():
                            print(f"✅ Selected: {custom_folder}")
                            return custom_folder
                        else:
                            print("❌ Folder not found or not accessible")
                    continue

                elif choice_num == len(icloud_folders) + 2:
                    print("❌ Cancelled")
                    return None

                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(icloud_folders)+2}")

            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None

    else:
        # No iCloud folders detected - manual entry
        print("🔍 No iCloud Drive folders detected")
        custom_path = input("📁 Enter full path to folder to backup: ").strip()

        if custom_path:
            custom_folder = Path(custom_path)
            if custom_folder.exists() and custom_folder.is_dir():
                return custom_folder
            else:
                print("❌ Folder not found or not accessible")

        return None

def show_backup_menu():
    """Show main backup menu"""

    if not BACKUP_AVAILABLE:
        print("❌ Backup system not available")
        return

    backup_manager = iCloudBackupManager()

    while True:
        print("\n🎛️  BACKUP MENU:")
        print("-" * 30)
        print("1. 💾 Create New Backup")
        print("2. 📂 List Existing Backups")
        print("3. 🔄 Restore Backup")
        print("4. ❌ Exit")

        try:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                # Create backup
                folder = select_folder_interactive()
                if folder:
                    create_backup_interactive(backup_manager, folder)

            elif choice == '2':
                # List backups
                list_backups_interactive(backup_manager)

            elif choice == '3':
                # Restore backup
                restore_backup_interactive(backup_manager)

            elif choice == '4':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-4")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def create_backup_interactive(backup_manager: iCloudBackupManager, source_folder: Path):
    """Interactive backup creation"""

    print(f"\n💾 CREATING BACKUP:")
    print(f"📂 Source: {source_folder}")

    # Check if it's iCloud
    icloud_path = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs'
    is_icloud = str(source_folder).startswith(str(icloud_path))

    if is_icloud:
        print("☁️  iCloud Drive folder detected - optimized backup will be used")

    # Get backup name
    default_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_name = input(f"💾 Backup name (default: {default_name}): ").strip()

    if not backup_name:
        backup_name = default_name

    # Confirm
    print(f"\n📋 BACKUP SUMMARY:")
    print(f"   📂 Source: {source_folder}")
    print(f"   💾 Name: {backup_name}")
    print(f"   ☁️  iCloud: {'Yes' if is_icloud else 'No'}")

    if input("\n✅ Proceed with backup? (y/n): ").lower().strip() != 'y':
        print("❌ Backup cancelled")
        return

    # Create backup with progress
    print(f"\n🔄 Creating backup...")
    progress_callback = create_progress_callback()

    success, backup_path, backup_info = backup_manager.create_backup(
        source_folder, backup_name, progress_callback
    )

    if success:
        print(f"\n✅ Backup completed successfully!")
        print(f"📁 Backup file: {backup_path}")
        print(f"📊 Size: {backup_info.get('backup_size_mb', 0):.1f} MB")

        if is_icloud and backup_info.get('downloaded', 0) > 0:
            print(f"☁️  Downloaded {backup_info['downloaded']} iCloud files")

    else:
        print(f"\n❌ Backup failed: {backup_path}")

def list_backups_interactive(backup_manager: iCloudBackupManager):
    """Interactive backup listing"""

    backups = backup_manager.list_backups()

    if not backups:
        print("\n📂 No backups found")
        return

    print(f"\n📂 AVAILABLE BACKUPS ({len(backups)} found):")
    print("="*80)

    for i, backup in enumerate(backups, 1):
        print(f"\n{i:2d}. 📦 {backup['name']}")
        print(f"     📁 File: {backup['file']}")
        print(f"     📊 Size: {backup['size_mb']:.1f} MB")
        print(f"     📅 Created: {backup['created'].strftime('%Y-%m-%d %H:%M:%S')}")

        if backup.get('source_directory'):
            print(f"     📂 Source: {backup['source_directory']}")

        if backup.get('total_files'):
            print(f"     📄 Files: {backup['total_files']}")

def restore_backup_interactive(backup_manager: iCloudBackupManager):
    """Interactive backup restore"""

    backups = backup_manager.list_backups()

    if not backups:
        print("\n📂 No backups found to restore")
        return

    print(f"\n🔄 SELECT BACKUP TO RESTORE:")
    print("-" * 50)

    for i, backup in enumerate(backups, 1):
        print(f"{i:2d}. {backup['name']} ({backup['size_mb']:.1f} MB) - {backup['created'].strftime('%Y-%m-%d %H:%M')}")

    print(f"{len(backups)+1:2d}. ❌ Cancel")

    try:
        choice = int(input(f"\nSelect backup (1-{len(backups)+1}): ").strip())

        if choice == len(backups) + 1:
            print("❌ Cancelled")
            return

        if not (1 <= choice <= len(backups)):
            print("❌ Invalid choice")
            return

        selected_backup = backups[choice - 1]
        backup_file = Path(selected_backup['file'])

        # Get restore location
        restore_path = input(f"\n📂 Restore to directory: ").strip()

        if not restore_path:
            print("❌ No restore path provided")
            return

        restore_dir = Path(restore_path)

        # Confirm
        print(f"\n📋 RESTORE SUMMARY:")
        print(f"   📦 Backup: {selected_backup['name']}")
        print(f"   📂 Restore to: {restore_dir}")

        if restore_dir.exists() and any(restore_dir.iterdir()):
            print(f"   ⚠️  Warning: Restore directory is not empty!")

        if input("\n✅ Proceed with restore? (y/n): ").lower().strip() != 'y':
            print("❌ Restore cancelled")
            return

        # Restore with progress
        print(f"\n🔄 Restoring backup...")
        progress_callback = create_progress_callback()

        success, result = backup_manager.restore_backup(backup_file, restore_dir, progress_callback)

        if success:
            print(f"\n✅ Restore completed successfully!")
            print(f"📂 Files restored to: {result}")
        else:
            print(f"\n❌ Restore failed: {result}")

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Cancelled")

def main():
    """Main function"""

    print_header()

    if not BACKUP_AVAILABLE:
        print("❌ Backup system not available. Please check installation.")
        return 1

    try:
        show_backup_menu()
        return 0

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0

if __name__ == '__main__':
    sys.exit(main())