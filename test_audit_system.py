#!/usr/bin/env python3
"""
🧪 AUDIT SYSTEM TEST SCRIPT
Quick validation of comprehensive audit functionality
"""

import os
import sys
import shutil
from pathlib import Path
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_structure():
    """Create test directory structure with sample files"""

    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix='audit_test_'))
    print(f"📁 Creating test structure in: {test_dir}")

    # Create folder structure
    folders = [
        'Container 1/Queena',
        'Container 2/Argy',
        'Container 3',
        'Argyrhs',  # Problematic folder
        'TEMPORARY',  # Problematic folder
        'Shared Documents'  # Problematic folder
    ]

    for folder in folders:
        (test_dir / folder).mkdir(parents=True, exist_ok=True)

    # Create sample files
    test_files = [
        # Container 1 - Complete set
        ('Container 1/Queena/Invoice_Queena_001.pdf', 'Sample invoice content'),
        ('Container 1/Queena/CE_Certificate_Gaming_Controller.pdf', 'CE certificate content'),
        ('Container 1/Queena/Packing_List_Container1.xlsx', 'Packing list content'),
        ('Container 1/Queena/BL_Container1.pdf', 'Bill of lading content'),

        # Container 2 - Missing CE
        ('Container 2/Argy/Invoice_Argy_002.pdf', 'Another invoice'),
        ('Container 2/Argy/Packing_List_Container2.xlsx', 'Packing list 2'),

        # Container 3 - Orphan files
        ('Container 3/CE_Random_Product.pdf', 'Orphan CE certificate'),
        ('Container 3/Manual_Gaming_Headset.pdf', 'Product manual'),

        # Problematic files
        ('Argyrhs/Container 2 Invoice.pdf', 'Misplaced invoice'),  # Wrong container reference
        ('TEMPORARY/Copy of Invoice.pdf', 'Temporary file'),
        ('TEMPORARY/Draft_Document.docx', 'Draft document'),
        ('Shared Documents/Send_to_Dad.pdf', 'File for parents'),  # Parent-sent pattern

        # Duplicates
        ('Container 1/Queena/Invoice_Queena_001_COPY.pdf', 'Sample invoice content'),  # Duplicate content
        ('Container 1/Queena/FINAL_Invoice_Queena_001.pdf', 'Sample invoice content'),  # Another duplicate

        # Bad filenames
        ('Container 1/Queena/Container 1.pdf', 'Bad filename - just container name'),
        ('Container 1/Queena/Doc1.pdf', 'Bad filename - generic name'),
        ('Container 1/Queena/Version_2.xlsx', 'Bad filename - version pattern'),
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

def test_audit_engine():
    """Test the audit engine functionality"""

    try:
        from advanced_audit_system import ComprehensiveAuditEngine
        from excel_report_generator import ExcelReportGenerator

        print("✅ Successfully imported audit modules")
        return True

    except ImportError as e:
        print(f"❌ Failed to import audit modules: {e}")
        return False

def run_test_audit(test_dir: Path):
    """Run audit on test directory"""

    try:
        from advanced_audit_system import ComprehensiveAuditEngine
        from excel_report_generator import ExcelReportGenerator

        # Create config for testing
        config = {
            'log_level': 'DEBUG',
            'enable_ai': False,  # Disable AI for faster testing
            'enable_ocr': False,
            'similarity_threshold': 0.85,
            'zero_change_mode': True
        }

        print("🔧 Initializing audit engine...")
        audit_engine = ComprehensiveAuditEngine(config)

        print("🔍 Running audit on test directory...")
        audit_results = audit_engine.run_comprehensive_audit(str(test_dir))

        # Generate test report
        report_generator = ExcelReportGenerator()
        output_file = 'test_audit_report.xlsx'

        print("📊 Generating test report...")
        report_file = report_generator.generate_comprehensive_report(
            audit_results,
            output_file
        )

        # Print summary
        summary = audit_results.get('summary', {})
        print(f"\n📈 TEST RESULTS:")
        print(f"   Files Analyzed: {summary.get('total_files', 0)}")
        print(f"   Issues Found: {summary.get('issues_found', 0)}")
        print(f"   Processing Time: {summary.get('processing_time', 0):.2f}s")

        print(f"✅ Test completed successfully!")
        print(f"📄 Report saved: {report_file}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""

    print("🧪 STARTING AUDIT SYSTEM TESTS")
    print("=" * 50)

    # Test 1: Module imports
    print("\n🔧 Test 1: Testing module imports...")
    if not test_audit_engine():
        print("❌ Module import test failed")
        return 1

    # Test 2: Create test structure
    print("\n📁 Test 2: Creating test directory structure...")
    test_dir = create_test_structure()

    try:
        # Test 3: Run audit
        print("\n🔍 Test 3: Running comprehensive audit...")
        if not run_test_audit(test_dir):
            print("❌ Audit test failed")
            return 1

        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Audit system is ready for use!")

        # Keep test files for inspection
        keep_files = input(f"\n📂 Keep test files in {test_dir} for inspection? (y/n): ").lower().strip()
        if keep_files not in ['y', 'yes']:
            shutil.rmtree(test_dir)
            print("🗑️  Test files cleaned up")
        else:
            print(f"📂 Test files preserved in: {test_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 130

    finally:
        # Cleanup if needed - only if keep_files was set
        try:
            if 'keep_files' in locals() and test_dir.exists() and keep_files not in ['y', 'yes']:
                shutil.rmtree(test_dir)
        except:
            pass

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)