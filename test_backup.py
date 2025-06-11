#!/usr/bin/env python3
"""
🧪 BACKUP SYSTEM TESTS
Test the backup functionality
"""

import sys
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create test data for backup testing"""

    # Create temporary directory with test files
    test_dir = Path(tempfile.mkdtemp(prefix="backup_test_"))

    # Create some test files
    (test_dir / "test1.txt").write_text("Test file 1", encoding='utf-8')
    (test_dir / "test2.pdf").write_text("Fake PDF content", encoding='utf-8')

    # Create subdirectory
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "test3.docx").write_text("Fake DOCX content", encoding='utf-8')

    print(f"📁 Created test data in: {test_dir}")
    return test_dir

def test_backup_system():
    """Test the backup system"""

    try:
        from icloud_backup_manager import iCloudBackupManager, create_progress_callback
        print("✅ Backup manager imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import backup manager: {e}")
        return False

    # Create test data
    test_data_dir = create_test_data()

    try:
        # Initialize backup manager
        backup_manager = iCloudBackupManager()
        print("✅ Backup manager initialized")

        # Test backup creation
        print("\n🔄 Testing backup creation...")
        progress_callback = create_progress_callback()

        success, backup_path, backup_info = backup_manager.create_backup(
            test_data_dir, "test_backup", progress_callback
        )

        if success:
            print(f"\n✅ Backup created: {backup_path}")
            print(f"📊 Size: {backup_info.get('backup_size_mb', 0):.2f} MB")

            # Test backup listing
            print("\n📂 Testing backup listing...")
            backups = backup_manager.list_backups()
            print(f"✅ Found {len(backups)} backups")

            # Test restore
            print("\n🔄 Testing restore...")
            restore_dir = Path(tempfile.mkdtemp(prefix="restore_test_"))

            restore_success, restore_result = backup_manager.restore_backup(
                Path(backup_path), restore_dir, progress_callback
            )

            if restore_success:
                print(f"✅ Restore successful: {restore_result}")

                # Verify restore
                restored_files = list(restore_dir.rglob('*'))
                print(f"📄 Restored {len(restored_files)} items")

                return True
            else:
                print(f"❌ Restore failed: {restore_result}")
                return False
        else:
            print(f"❌ Backup failed: {backup_path}")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    finally:
        # Cleanup
        try:
            shutil.rmtree(test_data_dir)
            print(f"🧹 Cleaned up test data")
        except:
            pass

def main():
    """Main test function"""

    print("🧪 TESTING BACKUP SYSTEM")
    print("=" * 50)

    if test_backup_system():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
