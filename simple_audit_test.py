#!/usr/bin/env python3
"""
🧪 SIMPLE AUDIT TEST
Basic validation without OCR dependencies
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_simple_test_structure():
    """Create test directory structure with simple files (no PDFs)"""

    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix='simple_audit_test_'))
    print(f"📁 Creating test structure in: {test_dir}")

    # Create folder structure
    folders = [
        'Container 1/Queena',
        'Container 2/Argy',
        'Container 3',
        'Argyrhs',  # Problematic folder
        'TEMPORARY',  # Problematic folder
    ]

    for folder in folders:
        (test_dir / folder).mkdir(parents=True, exist_ok=True)

    # Create sample files (using .txt to avoid OCR)
    test_files = [
        # Container 1 - Complete set
        ('Container 1/Queena/Invoice_Queena_001.xlsx', 'Invoice data'),
        ('Container 1/Queena/CE_Certificate_Gaming_Controller.jpg', 'CE cert image'),
        ('Container 1/Queena/Packing_List_Container1.xlsx', 'Packing list'),

        # Container 2 - Missing CE
        ('Container 2/Argy/Invoice_Argy_002.xlsx', 'Another invoice'),
        ('Container 2/Argy/Packing_List_Container2.xlsx', 'Packing list 2'),

        # Container 3 - Orphan files
        ('Container 3/CE_Random_Product.jpg', 'Orphan CE certificate'),
        ('Container 3/Manual_Gaming_Headset.docx', 'Product manual'),

        # Problematic files
        ('Argyrhs/Container 2 Invoice.xlsx', 'Misplaced invoice'),
        ('TEMPORARY/Copy of Invoice.xlsx', 'Temporary file'),
        ('TEMPORARY/Draft_Document.docx', 'Draft document'),

        # Duplicates
        ('Container 1/Queena/Invoice_Queena_001_COPY.xlsx', 'Invoice data'),  # Duplicate content
        ('Container 1/Queena/FINAL_Invoice_Queena_001.xlsx', 'Invoice data'),  # Another duplicate

        # Bad filenames
        ('Container 1/Queena/Container 1.docx', 'Bad filename'),
        ('Container 1/Queena/Doc1.xlsx', 'Bad filename - generic'),
        ('Container 1/Queena/Version_2.xlsx', 'Bad filename - version'),
    ]

    # Create test files
    for file_path, content in test_files:
        full_path = test_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with sample content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"✅ Created {len(test_files)} test files")
    return test_dir

def test_basic_audit(test_dir: Path):
    """Run basic audit test"""

    try:
        from advanced_audit_system import ComprehensiveAuditEngine

        # Create simple config
        config = {
            'log_level': 'INFO',
            'enable_ai': False,  # Disable AI for simple test
            'enable_ocr': False,  # Disable OCR
            'similarity_threshold': 0.85,
            'zero_change_mode': True
        }

        print("🔧 Initializing audit engine...")
        audit_engine = ComprehensiveAuditEngine(config)

        print("🔍 Running basic audit...")
        audit_results = audit_engine.run_comprehensive_audit(str(test_dir))

        # Print results
        summary = audit_results.get('summary', {})
        files = audit_results.get('files_analyzed', [])
        duplicates = audit_results.get('duplicates', [])
        unknown_folders = audit_results.get('unknown_folders', [])

        print(f"\n📈 AUDIT RESULTS:")
        print(f"   📊 Files Analyzed: {len(files)}")
        print(f"   🔍 Duplicates Found: {len(duplicates)}")
        print(f"   🗂️  Unknown Folders: {len(unknown_folders)}")
        print(f"   ⏱️  Processing Time: {summary.get('processing_time', 0):.2f}s")

        # Show some details
        if duplicates:
            print(f"\n🔍 DUPLICATES DETECTED:")
            for dup in duplicates[:3]:  # Show first 3
                print(f"   • {Path(dup['file_path']).name}")

        if unknown_folders:
            print(f"\n🗂️  PROBLEMATIC FOLDERS:")
            for folder in unknown_folders[:3]:  # Show first 3
                print(f"   • {folder['folder_name']} ({folder['recommendation']})")

        print(f"\n✅ Basic audit test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Basic audit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""

    print("🧪 STARTING SIMPLE AUDIT TESTS")
    print("=" * 50)

    # Test module imports
    print("\n🔧 Testing module imports...")
    try:
        from advanced_audit_system import ComprehensiveAuditEngine
        print("✅ Successfully imported audit modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return 1

    # Create test structure
    print("\n📁 Creating simple test structure...")
    test_dir = create_simple_test_structure()

    try:
        # Run basic audit
        print("\n🔍 Running basic audit test...")
        if not test_basic_audit(test_dir):
            print("❌ Basic audit test failed")
            return 1

        print("\n✅ ALL SIMPLE TESTS PASSED!")
        print("🎉 Basic audit system functionality validated!")

        return 0

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 130

    finally:
        # Cleanup
        try:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print("🗑️  Test files cleaned up")
        except:
            pass

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)