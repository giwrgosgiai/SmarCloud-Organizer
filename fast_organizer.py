#!/usr/bin/env python3
"""
🚀 FAST FILE ORGANIZER - Llama Consistency Checking
No heavy AI models - Direct Llama API calls only!
"""

import os
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

class FastFileOrganizer:
    def __init__(self):
        print("🚀 FAST ORGANIZER - Starting up...")
        self.processed_files = []

    def classify_with_llama(self, text_content: str, folder_path: str) -> Dict:
        """🤖 Direct Llama classification - no model loading!"""
        try:
            prompt = f"""
            Analyze this document:

            CONTENT: {text_content[:1000]}...
            FOLDER: {folder_path}

            Respond with: TYPE|CONFIDENCE|CONSISTENCY|ISSUES
            Example: Invoice|0.9|CONSISTENT|None
            """

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2:3b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1, 'max_tokens': 100}
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                llama_response = result.get('response', '').strip()

                if '|' in llama_response:
                    parts = llama_response.split('|')
                    return {
                        'doc_type': parts[0].strip() if len(parts) > 0 else 'Unknown',
                        'confidence': float(parts[1].strip()) if len(parts) > 1 and parts[1].replace('.','').isdigit() else 0.5,
                        'consistency': parts[2].strip() if len(parts) > 2 else 'UNKNOWN',
                        'issues': parts[3].strip() if len(parts) > 3 and parts[3].lower() != 'none' else None,
                        'llama_used': True
                    }

        except Exception as e:
            print(f"⚠️ Llama error: {e}")

        # Fallback classification
        return self.fallback_classify(text_content, folder_path)

    def fallback_classify(self, text_content: str, folder_path: str) -> Dict:
        """Simple pattern matching fallback"""
        text_lower = text_content.lower()
        folder_lower = folder_path.lower()

        # Document type detection
        if any(word in text_lower for word in ['invoice', 'τιμολογ', 'proforma']):
            doc_type = 'Invoice'
        elif any(word in text_lower for word in ['certificate', 'πιστοποιητικ', 'ce']):
            doc_type = 'Certificate'
        elif any(word in text_lower for word in ['manual', 'εγχειριδι', 'instruction']):
            doc_type = 'Manual'
        elif any(word in text_lower for word in ['packing', 'package', 'συσκευασ']):
            doc_type = 'Packing List'
        else:
            doc_type = 'Unknown'

        # Consistency check
        consistency_issues = []

        # Check for container mismatches
        import re
        content_containers = re.findall(r'container\s*(\d+)', text_lower)
        folder_containers = re.findall(r'container\s*(\d+)', folder_lower)

        if content_containers and folder_containers:
            if not set(content_containers).intersection(set(folder_containers)):
                consistency_issues.append(f"Container mismatch: Content mentions {content_containers} but folder is {folder_containers}")

        return {
            'doc_type': doc_type,
            'confidence': 0.7,
            'consistency': 'INCONSISTENT' if consistency_issues else 'CONSISTENT',
            'issues': '; '.join(consistency_issues) if consistency_issues else None,
            'llama_used': False
        }

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"⚠️ PDF extraction error for {pdf_path.name}: {e}")
            return ""

    def process_file(self, file_path: Path) -> Dict:
        """Process a single file"""
        try:
            stat = file_path.stat()
            folder_path = str(file_path.parent)

            print(f"📄 Processing: {file_path.name}")

            # Extract text for PDF files
            text_content = ""
            if file_path.suffix.lower() == '.pdf':
                text_content = self.extract_pdf_text(file_path)
                print(f"   📝 Extracted {len(text_content)} characters")

            # Classify with Llama
            if text_content and len(text_content.strip()) > 50:
                classification = self.classify_with_llama(text_content, folder_path)
                print(f"   🤖 {'Llama' if classification['llama_used'] else 'Pattern'} classified as: {classification['doc_type']} ({classification['confidence']:.2f})")

                if classification['issues']:
                    print(f"   ⚠️ CONSISTENCY ISSUES: {classification['issues']}")
            else:
                classification = {
                    'doc_type': 'No Content',
                    'confidence': 0.1,
                    'consistency': 'UNKNOWN',
                    'issues': None,
                    'llama_used': False
                }

            return {
                'File_Name': file_path.name,
                'Full_Path': str(file_path),
                'Folder_Path': folder_path,
                'File_Size_MB': round(stat.st_size / (1024*1024), 3),
                'Modified_Date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'Document_Type': classification['doc_type'],
                'Confidence': classification['confidence'],
                'Consistency_Check': classification['consistency'],
                'Consistency_Issues': classification['issues'] or '',
                'Text_Length': len(text_content),
                'AI_Method': 'Llama' if classification['llama_used'] else 'Pattern',
                'Content_Sample': text_content[:200] + "..." if len(text_content) > 200 else text_content
            }

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return {
                'File_Name': file_path.name,
                'Full_Path': str(file_path),
                'Document_Type': 'ERROR',
                'Error': str(e)
            }

    def scan_folder(self, folder_path: str) -> List[Dict]:
        """Scan folder and process files"""
        print(f"🔍 Scanning folder: {folder_path}")

        base_path = Path(folder_path)
        if not base_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return []

        # Find files to process
        files_to_process = []
        for file_path in base_path.rglob("*"):
            if (file_path.is_file() and
                file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt'] and
                file_path.stat().st_size > 1000):  # Skip tiny files
                files_to_process.append(file_path)

        print(f"📊 Found {len(files_to_process)} files to process")

        # Process files
        results = []
        for i, file_path in enumerate(files_to_process, 1):
            print(f"\n[{i}/{len(files_to_process)}]", end=" ")
            result = self.process_file(file_path)
            results.append(result)

            # Progress update
            if i % 5 == 0:
                print(f"\n✅ Processed {i}/{len(files_to_process)} files")

        return results

    def export_to_excel(self, results: List[Dict], output_file: str):
        """Export results to Excel"""
        print(f"\n📊 Exporting to {output_file}...")

        if not results:
            print("❌ No results to export")
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add summary statistics
        total_files = len(df)
        llama_used = len(df[df.get('AI_Method', '') == 'Llama'])
        consistency_issues = len(df[df.get('Consistency_Issues', '') != ''])

        print(f"📈 SUMMARY:")
        print(f"   📁 Total files: {total_files}")
        print(f"   🤖 Llama classifications: {llama_used}")
        print(f"   ⚠️ Consistency issues found: {consistency_issues}")

        # Export to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='File Analysis', index=False)

            # Summary sheet
            summary_data = {
                'Metric': ['Total Files', 'Llama Classifications', 'Consistency Issues', 'Processing Time'],
                'Value': [total_files, llama_used, consistency_issues, f"{time.time():.1f}s"]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        print(f"✅ Exported to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fast_organizer.py <folder_path> [output_file.xlsx]")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "fast_analysis.xlsx"

    print("🚀 FAST FILE ORGANIZER with Llama Consistency Checking")
    print("=" * 60)

    start_time = time.time()

    # Initialize organizer
    organizer = FastFileOrganizer()

    # Process files
    results = organizer.scan_folder(folder_path)

    # Export results
    if results:
        organizer.export_to_excel(results, output_file)

    elapsed = time.time() - start_time
    print(f"\n🎉 Completed in {elapsed:.1f} seconds!")

if __name__ == "__main__":
    main()