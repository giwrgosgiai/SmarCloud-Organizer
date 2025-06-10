#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE AUDIT RUNNER
🎯 Zero-change audit system with AI integration
🇬🇷 Full Greek support with all requested features

This script implements all 13 audit requirements:
1. Unknown folder detection
2. Duplicate file detection (fuzzy + content)
3. Bad filename detection
4. Shipping date extraction (OCR)
5. Customs cost calculation
6. Orphan CE detection
7. Shared CE detection
8. Missing document analysis
9. Parent-sent file detection
10. Interactive UI with progress tracking
11. Zero-change philosophy
12. AI integration for advanced analysis
13. Comprehensive Excel reporting with dropdowns
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from advanced_audit_system import ComprehensiveAuditEngine
    from excel_report_generator import ExcelReportGenerator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    MODULES_AVAILABLE = False

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup comprehensive logging"""

    # Create logs directory if it doesn't exist
    logs_dir = Path('audit_logs')
    logs_dir.mkdir(exist_ok=True)

    # Setup logger
    logger = logging.getLogger('comprehensive_audit')
    logger.setLevel(getattr(logging, log_level.upper()))

    # File handler
    file_handler = logging.FileHandler(logs_dir / f'audit_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def create_default_config() -> Dict:
    """Create default configuration for audit"""
    return {
        'log_level': 'INFO',
        'enable_ai': True,
        'enable_ocr': True,
        'similarity_threshold': 0.85,
        'max_threads': 4,
        'output_format': 'xlsx',
        'backup_before_changes': True,
        'zero_change_mode': True,
        'target_extensions': ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.jpg', '.jpeg', '.png'],
        'problematic_folders': [
            'argyrhs', 'anna', 'other company', 'frankouli',
            'shared', 'tzika', 'deleted', 'temporary', 'temp',
            'copy', 'old', 'backup', 'archive'
        ],
        'known_suppliers': ['queena', 'argy', 'anna', 'frankouli'],
        'required_documents': ['invoice', 'ce', 'packing_list'],
        'optional_documents': ['bl', 'manual', 'bank_proof']
    }

def print_welcome_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🔍 COMPREHENSIVE FILE AUDIT SYSTEM v2.0                                   ║
║    🏢 Enterprise-grade analysis with AI integration                          ║
║    🇬🇷 Full Greek language support                                          ║
║                                                                              ║
║    Features:                                                                 ║
║    • 🤖 AI-powered classification and consistency checking                   ║
║    • 📊 Comprehensive duplicate detection (content + fuzzy names)           ║
║    • 🗂️ Unknown folder identification and recommendations                   ║
║    • 📋 Document completeness analysis per container                         ║
║    • 🎯 Zero-change philosophy - audit only, no modifications               ║
║    • 📈 Interactive Excel reports with dropdown menus                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_target_directory(target_dir: str) -> bool:
    """Validate that target directory exists and is accessible"""
    target_path = Path(target_dir)

    if not target_path.exists():
        print(f"❌ Error: Directory '{target_dir}' does not exist")
        return False

    if not target_path.is_dir():
        print(f"❌ Error: '{target_dir}' is not a directory")
        return False

    if not os.access(target_path, os.R_OK):
        print(f"❌ Error: No read access to directory '{target_dir}'")
        return False

    # Count relevant files
    file_count = 0
    for file_path in target_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.jpg', '.jpeg', '.png']:
            file_count += 1

    print(f"📁 Directory validated: {target_dir}")
    print(f"📊 Found {file_count} relevant files for analysis")

    if file_count == 0:
        print("⚠️  Warning: No relevant files found in target directory")
        return False

    return True

def get_user_confirmation(message: str) -> bool:
    """Get user confirmation before proceeding"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

def main():
    """Main function to run comprehensive audit"""

    # Print welcome banner
    print_welcome_banner()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Comprehensive File Audit System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_audit.py --target "/path/to/business/documents"
  python run_comprehensive_audit.py --target "/path/to/docs" --output "custom_audit.xlsx"
  python run_comprehensive_audit.py --target "/path/to/docs" --no-ai --verbose
        """
    )

    parser.add_argument(
        '--target', '-t',
        required=True,
        help='Target directory to audit'
    )

    parser.add_argument(
        '--output', '-o',
        default='comprehensive_audit_report.xlsx',
        help='Output Excel file name (default: comprehensive_audit_report.xlsx)'
    )

    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (JSON format)'
    )

    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Disable AI features for faster processing'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform audit without generating report (for testing)'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)

    logger.info("🚀 Starting Comprehensive Audit System")

    # Check if modules are available
    if not MODULES_AVAILABLE:
        logger.error("❌ Required modules not available. Please check imports.")
        return 1

    # Validate target directory
    if not validate_target_directory(args.target):
        logger.error("❌ Target directory validation failed")
        return 1

    # Load configuration
    if args.config and Path(args.config).exists():
        logger.info(f"📄 Loading configuration from: {args.config}")
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️  Failed to load config file: {e}. Using defaults.")
            config = create_default_config()
    else:
        logger.info("📄 Using default configuration")
        config = create_default_config()

    # Override config with CLI arguments
    if args.no_ai:
        config['enable_ai'] = False
        logger.info("🤖 AI features disabled by user request")

    if args.verbose:
        config['log_level'] = 'DEBUG'

    # Print configuration summary
    print("\n📋 AUDIT CONFIGURATION:")
    print(f"   Target Directory: {args.target}")
    print(f"   Output File: {args.output}")
    print(f"   AI Enabled: {'✅' if config['enable_ai'] else '❌'}")
    print(f"   OCR Enabled: {'✅' if config['enable_ocr'] else '❌'}")
    print(f"   Zero-Change Mode: {'✅' if config['zero_change_mode'] else '❌'}")
    print(f"   Similarity Threshold: {config['similarity_threshold']}")

    # Get user confirmation
    if not get_user_confirmation("\n🔍 Proceed with comprehensive audit?"):
        logger.info("🛑 Audit cancelled by user")
        return 0

    try:
        # Initialize audit engine
        logger.info("🔧 Initializing Comprehensive Audit Engine...")
        audit_engine = ComprehensiveAuditEngine(config)

        # Run comprehensive audit
        logger.info("🔍 Starting comprehensive audit analysis...")
        print("\n" + "="*80)
        print("🔍 RUNNING COMPREHENSIVE AUDIT")
        print("="*80)

        audit_results = audit_engine.run_comprehensive_audit(args.target)

        if not args.dry_run:
            # Generate Excel report
            logger.info("📊 Generating comprehensive Excel report...")
            report_generator = ExcelReportGenerator()

            # Ensure output file has .xlsx extension
            output_file = args.output
            if not output_file.endswith('.xlsx'):
                output_file += '.xlsx'

            report_file = report_generator.generate_comprehensive_report(
                audit_results,
                output_file
            )

            # Print summary
            print("\n" + "="*80)
            print("✅ AUDIT COMPLETED SUCCESSFULLY")
            print("="*80)

            summary = audit_results.get('summary', {})
            print(f"📊 Total Files Analyzed: {summary.get('total_files', 0)}")
            print(f"⚠️  Issues Found: {summary.get('issues_found', 0)}")
            print(f"🤖 AI Success Rate: {summary.get('ai_success_rate', 0):.1f}%")
            print(f"⏱️  Processing Time: {summary.get('processing_time', 0):.2f} seconds")
            print(f"📈 Files per Second: {summary.get('files_per_second', 0):.2f}")

            print(f"\n📄 Comprehensive report saved: {report_file}")
            print(f"📂 Log files saved in: ./audit_logs/")

            # Recommendations
            print("\n💡 NEXT STEPS:")
            print("   1. Review the Excel report with all audit findings")
            print("   2. Use dropdown menus in Excel for action selection")
            print("   3. Start with HIGH PRIORITY items first")
            print("   4. Remember: This audit made NO changes to your files")
            print("   5. Create backups before implementing any recommendations")

        else:
            print("\n🧪 DRY RUN COMPLETED - No report generated")

        logger.info("✅ Comprehensive audit completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Audit interrupted by user")
        print("\n🛑 Audit interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"❌ Audit failed with error: {e}", exc_info=True)
        print(f"\n❌ Audit failed: {e}")
        print("📂 Check log files in ./audit_logs/ for detailed error information")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)