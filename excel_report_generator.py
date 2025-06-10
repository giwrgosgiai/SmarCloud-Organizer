#!/usr/bin/env python3
"""
📊 ADVANCED EXCEL REPORT GENERATOR
🏢 Enterprise-grade Excel reporting with interactive features
🇬🇷 Full Greek support with dropdown menus and conditional formatting
"""

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule, IconSetRule, CellIsRule, FormulaRule
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.chart import BarChart, PieChart, Reference
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
import datetime
from typing import Dict, List, Any
from pathlib import Path

class ExcelReportGenerator:
    """Advanced Excel report generator with interactive features"""

    def __init__(self):
        self.setup_styles()

    def setup_styles(self):
        """Setup predefined styles for Excel formatting"""
        self.styles = {
            'header': {
                'font': Font(name='Calibri', size=12, bold=True, color='FFFFFF'),
                'fill': PatternFill(start_color='366092', end_color='366092', fill_type='solid'),
                'alignment': Alignment(horizontal='center', vertical='center', wrap_text=True),
                'border': Border(
                    left=Side(border_style='thin'),
                    right=Side(border_style='thin'),
                    top=Side(border_style='thin'),
                    bottom=Side(border_style='thin')
                )
            },
            'data': {
                'font': Font(name='Calibri', size=10),
                'alignment': Alignment(horizontal='left', vertical='center', wrap_text=True),
                'border': Border(
                    left=Side(border_style='thin', color='D3D3D3'),
                    right=Side(border_style='thin', color='D3D3D3'),
                    top=Side(border_style='thin', color='D3D3D3'),
                    bottom=Side(border_style='thin', color='D3D3D3')
                )
            },
            'summary': {
                'font': Font(name='Calibri', size=11, bold=True),
                'fill': PatternFill(start_color='E6F3FF', end_color='E6F3FF', fill_type='solid'),
                'alignment': Alignment(horizontal='center', vertical='center')
            },
            'issue': {
                'font': Font(name='Calibri', size=10, color='CC0000'),
                'fill': PatternFill(start_color='FFE6E6', end_color='FFE6E6', fill_type='solid'),
                'alignment': Alignment(horizontal='left', vertical='center', wrap_text=True)
            },
            'success': {
                'font': Font(name='Calibri', size=10, color='006600'),
                'fill': PatternFill(start_color='E6FFE6', end_color='E6FFE6', fill_type='solid'),
                'alignment': Alignment(horizontal='left', vertical='center')
            }
        }

    def generate_comprehensive_report(self, audit_data: Dict, output_file: str):
        """Generate comprehensive Excel report with all audit results"""

        # Create workbook and sheets
        wb = openpyxl.Workbook()

        # Remove default sheet
        wb.remove(wb.active)

        # Create sheets
        summary_sheet = wb.create_sheet('📊 Σύνοψη Audit', 0)
        files_sheet = wb.create_sheet('📁 Ανάλυση Αρχείων', 1)
        issues_sheet = wb.create_sheet('⚠️ Προβλήματα', 2)
        folders_sheet = wb.create_sheet('🗂️ Άγνωστοι Φάκελοι', 3)
        duplicates_sheet = wb.create_sheet('🔍 Διπλά Αρχεία', 4)
        completeness_sheet = wb.create_sheet('📋 Πληρότητα', 5)
        recommendations_sheet = wb.create_sheet('💡 Προτάσεις', 6)

        # Generate each sheet
        self.create_summary_sheet(summary_sheet, audit_data)
        self.create_files_analysis_sheet(files_sheet, audit_data)
        self.create_issues_sheet(issues_sheet, audit_data)
        self.create_folders_sheet(folders_sheet, audit_data)
        self.create_duplicates_sheet(duplicates_sheet, audit_data)
        self.create_completeness_sheet(completeness_sheet, audit_data)
        self.create_recommendations_sheet(recommendations_sheet, audit_data)

        # Save workbook
        wb.save(output_file)
        print(f"✅ Comprehensive report generated: {output_file}")

        return output_file

    def create_summary_sheet(self, sheet, audit_data: Dict):
        """Create summary dashboard sheet"""

        # Title
        sheet['A1'] = '🔍 COMPREHENSIVE AUDIT SUMMARY'
        sheet['A1'].font = Font(name='Calibri', size=18, bold=True, color='366092')
        sheet.merge_cells('A1:H1')

        # Audit info
        row = 3
        sheet[f'A{row}'] = 'Audit Date:'
        sheet[f'B{row}'] = audit_data.get('audit_timestamp', '')
        sheet[f'A{row+1}'] = 'Total Files Analyzed:'
        sheet[f'B{row+1}'] = len(audit_data.get('files_analyzed', []))
        sheet[f'A{row+2}'] = 'Processing Time:'
        sheet[f'B{row+2}'] = f"{audit_data.get('summary', {}).get('processing_time', 0):.2f} seconds"

        # Key metrics
        row = 7
        metrics = [
            ('📊 Total Files', len(audit_data.get('files_analyzed', []))),
            ('⚠️ Issues Found', len(audit_data.get('consistency_issues', []))),
            ('🔍 Duplicates', len(audit_data.get('duplicates', []))),
            ('🗂️ Unknown Folders', len(audit_data.get('unknown_folders', []))),
            ('🤖 AI Classifications', audit_data.get('summary', {}).get('ai_success_rate', 0)),
            ('📋 Containers Analyzed', len(audit_data.get('completeness_analysis', {})))
        ]

        for i, (metric, value) in enumerate(metrics):
            sheet[f'A{row+i}'] = metric
            sheet[f'B{row+i}'] = value

            # Apply styling
            sheet[f'A{row+i}'].font = self.styles['summary']['font']
            sheet[f'A{row+i}'].fill = self.styles['summary']['fill']
            sheet[f'A{row+i}'].alignment = self.styles['summary']['alignment']
            sheet[f'B{row+i}'].font = self.styles['summary']['font']
            sheet[f'B{row+i}'].fill = self.styles['summary']['fill']
            sheet[f'B{row+i}'].alignment = self.styles['summary']['alignment']

        # Issues summary table
        row = 15
        sheet[f'A{row}'] = 'ISSUES BREAKDOWN'
        sheet[f'A{row}'].font = Font(bold=True, size=14)

        issues_data = [
            ['Issue Type', 'Count', 'Severity', 'Action Required'],
            ['Container Mismatches', len([i for i in audit_data.get('consistency_issues', []) if any('container' in issue.get('type', '') for issue in i.get('issues', []))]), 'High', 'Review & Move'],
            ['Duplicate Files', len(audit_data.get('duplicates', [])), 'Medium', 'Delete Copies'],
            ['Unknown Folders', len(audit_data.get('unknown_folders', [])), 'Low', 'Reorganize'],
            ['Missing Documents', sum(len(c.get('issues', [])) for c in audit_data.get('completeness_analysis', {}).values()), 'Medium', 'Source Missing Docs']
        ]

        for i, row_data in enumerate(issues_data):
            for j, cell_data in enumerate(row_data):
                cell = sheet[f'{chr(65+j)}{row+i+1}']
                cell.value = cell_data

                if i == 0:  # Header row
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
                else:
                    cell.font = Font(name='Calibri', size=10)

                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = Border(
                    left=Side(border_style='thin'),
                    right=Side(border_style='thin'),
                    top=Side(border_style='thin'),
                    bottom=Side(border_style='thin')
                )

        # Auto-adjust column widths
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            sheet.column_dimensions[column_letter].width = adjusted_width

    def create_files_analysis_sheet(self, sheet, audit_data: Dict):
        """Create detailed files analysis sheet with dropdowns"""

        files_data = audit_data.get('files_analyzed', [])

        if not files_data:
            sheet['A1'] = 'No files analyzed'
            return

        # Create headers based on user requirements
        headers = [
            'Supplier', 'Year', 'Container', 'Invoice', 'PI', 'CE %', 'CE Missing List',
            'Packing List', 'BL Found', 'Load Date', 'Arrival Date', 'Customs Cost',
            'Orphans', 'Shared CE', 'Duplicate Files', 'Unknown Folders', 'Notes', 'Status'
        ]

        # Add headers
        for i, header in enumerate(headers):
            cell = sheet[f'{chr(65+i)}1']
            cell.value = header
            cell.font = self.styles['header']['font']
            cell.fill = self.styles['header']['fill']
            cell.alignment = self.styles['header']['alignment']
            cell.border = self.styles['header']['border']

        # Process files and group by container
        containers = {}
        for file_info in files_data:
            container = file_info.get('container', 'Unknown')
            if container not in containers:
                containers[container] = {
                    'files': [],
                    'invoice_count': 0,
                    'ce_count': 0,
                    'pl_count': 0,
                    'bl_count': 0,
                    'total_files': 0
                }

            containers[container]['files'].append(file_info)
            containers[container]['total_files'] += 1

            doc_type = file_info.get('document_type', '').lower()
            if 'invoice' in doc_type:
                containers[container]['invoice_count'] += 1
            elif 'ce' in doc_type:
                containers[container]['ce_count'] += 1
            elif 'pack' in doc_type:
                containers[container]['pl_count'] += 1
            elif 'bl' in doc_type:
                containers[container]['bl_count'] += 1

        # Add data rows
        row = 2
        for container_name, container_data in containers.items():
            # Calculate CE coverage
            total_products = container_data['invoice_count']  # Approximation
            ce_coverage = (container_data['ce_count'] / max(total_products, 1)) * 100

            # Determine completeness status
            has_invoice = container_data['invoice_count'] > 0
            has_ce = container_data['ce_count'] > 0
            has_pl = container_data['pl_count'] > 0
            has_bl = container_data['bl_count'] > 0

            status = 'COMPLETE' if all([has_invoice, has_ce, has_pl]) else 'INCOMPLETE'

            # Extract supplier and year from first file
            first_file = container_data['files'][0] if container_data['files'] else {}
            supplier = first_file.get('supplier', 'Unknown')
            year = first_file.get('year', 'Unknown')

            # Fill row data
            row_data = [
                supplier,                                    # Supplier
                year,                                        # Year
                container_name,                              # Container
                '✅' if has_invoice else '❌',              # Invoice
                '✅' if has_invoice else '❌',              # PI (same as invoice for now)
                f'{ce_coverage:.1f}%',                       # CE %
                'Missing CE analysis' if ce_coverage < 100 else 'Complete',  # CE Missing List
                '✅' if has_pl else '❌',                   # Packing List
                '✅' if has_bl else '❌',                   # BL Found
                'Not extracted',                             # Load Date (would need OCR)
                'Not extracted',                             # Arrival Date (would need OCR)
                'Not calculated',                            # Customs Cost (would need OCR)
                'Analysis needed',                           # Orphans
                'Analysis needed',                           # Shared CE
                'No duplicates' if container_name not in [d.get('container', '') for d in audit_data.get('duplicates', [])] else 'Has duplicates',  # Duplicate Files
                'Organized' if container_name not in [f.get('folder_name', '') for f in audit_data.get('unknown_folders', [])] else 'Needs review',  # Unknown Folders
                f'Total files: {container_data["total_files"]}',  # Notes
                status                                       # Status
            ]

            for i, value in enumerate(row_data):
                cell = sheet[f'{chr(65+i)}{row}']
                cell.value = value

                # Apply styling based on content
                if value in ['❌', 'INCOMPLETE', 'Has duplicates']:
                    cell.font = self.styles['issue']['font']
                    cell.fill = self.styles['issue']['fill']
                elif value in ['✅', 'COMPLETE', 'No duplicates']:
                    cell.font = self.styles['success']['font']
                    cell.fill = self.styles['success']['fill']
                else:
                    cell.font = self.styles['data']['font']

                cell.alignment = self.styles['data']['alignment']
                cell.border = self.styles['data']['border']

            row += 1

        # Add dropdowns for Status column
        status_dropdown = DataValidation(
            type="list",
            formula1='"COMPLETE,INCOMPLETE,NEEDS_REVIEW,IN_PROGRESS"',
            allow_blank=False
        )
        status_dropdown.prompt = 'Select status'
        status_dropdown.promptTitle = 'Status Selection'

        status_col = chr(65 + len(headers) - 1)  # Last column (Status)
        sheet.add_data_validation(status_dropdown)
        status_dropdown.add(f'{status_col}2:{status_col}{row-1}')

        # Add action dropdown for recommendations
        action_dropdown = DataValidation(
            type="list",
            formula1='"KEEP,DELETE,REVIEW,MOVE,RENAME"',
            allow_blank=True
        )
        action_dropdown.prompt = 'Select recommended action'
        action_dropdown.promptTitle = 'Action Selection'

        # Add action column
        action_col = chr(65 + len(headers))
        sheet[f'{action_col}1'] = 'Recommended Action'
        sheet[f'{action_col}1'].font = self.styles['header']['font']
        sheet[f'{action_col}1'].fill = self.styles['header']['fill']

        sheet.add_data_validation(action_dropdown)
        action_dropdown.add(f'{action_col}2:{action_col}{row-1}')

        # Auto-adjust column widths
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            sheet.column_dimensions[column_letter].width = adjusted_width

    def create_issues_sheet(self, sheet, audit_data: Dict):
        """Create issues tracking sheet"""

        consistency_issues = audit_data.get('consistency_issues', [])

        headers = ['File Path', 'Issue Type', 'Description', 'Severity', 'Recommended Action', 'Status', 'Notes']

        # Add headers
        for i, header in enumerate(headers):
            cell = sheet[f'{chr(65+i)}1']
            cell.value = header
            cell.font = self.styles['header']['font']
            cell.fill = self.styles['header']['fill']
            cell.alignment = self.styles['header']['alignment']
            cell.border = self.styles['header']['border']

        # Add issue data
        row = 2
        for issue in consistency_issues:
            file_path = issue.get('file_path', '')

            for problem in issue.get('issues', []):
                row_data = [
                    file_path,
                    problem.get('type', 'Unknown'),
                    problem.get('description', ''),
                    problem.get('severity', 'Medium'),
                    issue.get('recommended_action', 'REVIEW'),
                    'OPEN',
                    f"Confidence: {issue.get('confidence', 0):.2f}"
                ]

                for i, value in enumerate(row_data):
                    cell = sheet[f'{chr(65+i)}{row}']
                    cell.value = value
                    cell.font = self.styles['data']['font']
                    cell.alignment = self.styles['data']['alignment']
                    cell.border = self.styles['data']['border']

                row += 1

        # Add dropdowns
        severity_dropdown = DataValidation(
            type="list",
            formula1='"High,Medium,Low"',
            allow_blank=False
        )
        sheet.add_data_validation(severity_dropdown)
        severity_dropdown.add(f'D2:D{row-1}')

        status_dropdown = DataValidation(
            type="list",
            formula1='"OPEN,IN_PROGRESS,RESOLVED,CLOSED"',
            allow_blank=False
        )
        sheet.add_data_validation(status_dropdown)
        status_dropdown.add(f'F2:F{row-1}')

        # Auto-adjust column widths
        self.auto_adjust_columns(sheet)

    def create_folders_sheet(self, sheet, audit_data: Dict):
        """Create unknown folders analysis sheet"""

        unknown_folders = audit_data.get('unknown_folders', [])

        headers = ['Folder Path', 'Folder Name', 'Is Problematic', 'File Count', 'Recommendation', 'Selected Action', 'Notes']

        # Add headers
        for i, header in enumerate(headers):
            cell = sheet[f'{chr(65+i)}1']
            cell.value = header
            cell.font = self.styles['header']['font']
            cell.fill = self.styles['header']['fill']
            cell.alignment = self.styles['header']['alignment']
            cell.border = self.styles['header']['border']

        # Add folder data
        row = 2
        for folder in unknown_folders:
            row_data = [
                folder.get('folder_path', ''),
                folder.get('folder_name', ''),
                '✅' if folder.get('is_problematic', False) else '❌',
                folder.get('file_count', 0),
                folder.get('recommendation', ''),
                '',  # Empty for user selection
                'Requires manual review'
            ]

            for i, value in enumerate(row_data):
                cell = sheet[f'{chr(65+i)}{row}']
                cell.value = value

                if folder.get('is_problematic', False) and i < 5:
                    cell.font = self.styles['issue']['font']
                    cell.fill = self.styles['issue']['fill']
                else:
                    cell.font = self.styles['data']['font']

                cell.alignment = self.styles['data']['alignment']
                cell.border = self.styles['data']['border']

            row += 1

        # Add action dropdown
        action_dropdown = DataValidation(
            type="list",
            formula1='"Ignore,Merge with supplier folder,Rename to standard format,Move contents to proper location"',
            allow_blank=True
        )
        sheet.add_data_validation(action_dropdown)
        action_dropdown.add(f'F2:F{row-1}')

        self.auto_adjust_columns(sheet)

    def create_duplicates_sheet(self, sheet, audit_data: Dict):
        """Create duplicates analysis sheet"""

        duplicates = audit_data.get('duplicates', [])

        headers = ['File Path', 'Duplicate Of', 'Duplicate Type', 'File Size', 'Modified Date', 'Recommendation', 'Action', 'Notes']

        # Add headers
        for i, header in enumerate(headers):
            cell = sheet[f'{chr(65+i)}1']
            cell.value = header
            cell.font = self.styles['header']['font']
            cell.fill = self.styles['header']['fill']
            cell.alignment = self.styles['header']['alignment']
            cell.border = self.styles['header']['border']

        # Add duplicate data
        row = 2
        for duplicate in duplicates:
            file_path = Path(duplicate.get('file_path', ''))

            # Get file info
            try:
                file_size = file_path.stat().st_size / 1024  # KB
                mod_date = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            except:
                file_size = 0
                mod_date = 'Unknown'

            row_data = [
                str(file_path),
                duplicate.get('duplicate_of', ''),
                duplicate.get('duplicate_type', 'exact_content'),
                f'{file_size:.1f} KB',
                mod_date,
                duplicate.get('recommendation', 'DELETE'),
                '',  # For user action
                'Detected duplicate content'
            ]

            for i, value in enumerate(row_data):
                cell = sheet[f'{chr(65+i)}{row}']
                cell.value = value
                cell.font = self.styles['data']['font']

                if duplicate.get('recommendation') == 'DELETE' and i == 5:
                    cell.fill = self.styles['issue']['fill']

                cell.alignment = self.styles['data']['alignment']
                cell.border = self.styles['data']['border']

            row += 1

        # Add action dropdown
        action_dropdown = DataValidation(
            type="list",
            formula1='"DELETE,KEEP,RENAME,MOVE_TO_BACKUP"',
            allow_blank=True
        )
        sheet.add_data_validation(action_dropdown)
        action_dropdown.add(f'G2:G{row-1}')

        self.auto_adjust_columns(sheet)

    def create_completeness_sheet(self, sheet, audit_data: Dict):
        """Create document completeness analysis sheet"""

        completeness = audit_data.get('completeness_analysis', {})

        headers = ['Container', 'Invoice Count', 'CE Count', 'Packing List', 'BL/HBL', 'Manual', 'Bank Proof', 'Completeness Score', 'Issues', 'Status']

        # Add headers
        for i, header in enumerate(headers):
            cell = sheet[f'{chr(65+i)}1']
            cell.value = header
            cell.font = self.styles['header']['font']
            cell.fill = self.styles['header']['fill']
            cell.alignment = self.styles['header']['alignment']
            cell.border = self.styles['header']['border']

        # Add completeness data
        row = 2
        for container, data in completeness.items():
            counts = data.get('document_counts', {})
            score = data.get('completeness_score', 0)
            issues = '; '.join(data.get('issues', []))

            status = 'COMPLETE' if score >= 80 else 'INCOMPLETE' if score >= 50 else 'CRITICAL'

            row_data = [
                container,
                counts.get('invoice', 0),
                counts.get('ce', 0),
                counts.get('packing_list', 0),
                counts.get('bl', 0),
                counts.get('manual', 0),
                counts.get('bank_proof', 0),
                f'{score:.1f}%',
                issues if issues else 'No issues',
                status
            ]

            for i, value in enumerate(row_data):
                cell = sheet[f'{chr(65+i)}{row}']
                cell.value = value

                # Color coding based on completeness score
                if status == 'CRITICAL':
                    cell.fill = PatternFill(start_color='FFE6E6', end_color='FFE6E6', fill_type='solid')
                elif status == 'INCOMPLETE':
                    cell.fill = PatternFill(start_color='FFFACD', end_color='FFFACD', fill_type='solid')
                else:
                    cell.fill = PatternFill(start_color='E6FFE6', end_color='E6FFE6', fill_type='solid')

                cell.font = self.styles['data']['font']
                cell.alignment = self.styles['data']['alignment']
                cell.border = self.styles['data']['border']

            row += 1

        self.auto_adjust_columns(sheet)

    def create_recommendations_sheet(self, sheet, audit_data: Dict):
        """Create actionable recommendations sheet"""

        # Title
        sheet['A1'] = '💡 ACTIONABLE RECOMMENDATIONS'
        sheet['A1'].font = Font(name='Calibri', size=16, bold=True, color='366092')
        sheet.merge_cells('A1:D1')

        row = 3

        # High priority recommendations
        recommendations = [
            {
                'category': 'HIGH PRIORITY',
                'items': [
                    'Review container mismatches - files may be in wrong folders',
                    'Delete duplicate files to save space and reduce confusion',
                    'Organize unknown/problematic folders',
                    'Source missing CE certificates for incomplete containers'
                ]
            },
            {
                'category': 'MEDIUM PRIORITY',
                'items': [
                    'Implement consistent naming convention for files',
                    'Create backup of important documents before reorganization',
                    'Extract shipping dates from BL/HBL documents',
                    'Calculate customs costs from available documents'
                ]
            },
            {
                'category': 'LOW PRIORITY',
                'items': [
                    'Archive old temporary files',
                    'Create document templates for future use',
                    'Implement automated filing system',
                    'Regular audit schedule (monthly/quarterly)'
                ]
            }
        ]

        for recommendation in recommendations:
            # Category header
            sheet[f'A{row}'] = recommendation['category']
            sheet[f'A{row}'].font = Font(name='Calibri', size=14, bold=True, color='CC0000' if 'HIGH' in recommendation['category'] else '996600' if 'MEDIUM' in recommendation['category'] else '006600')
            row += 1

            # Items
            for item in recommendation['items']:
                sheet[f'B{row}'] = f"• {item}"
                sheet[f'B{row}'].font = Font(name='Calibri', size=11)
                sheet[f'B{row}'].alignment = Alignment(wrap_text=True)
                row += 1

            row += 1  # Empty row between categories

        # Implementation checklist
        row += 2
        sheet[f'A{row}'] = 'IMPLEMENTATION CHECKLIST'
        sheet[f'A{row}'].font = Font(name='Calibri', size=14, bold=True)
        row += 1

        checklist_items = [
            'Backup all files before making changes',
            'Start with high-priority items',
            'Test changes on a small subset first',
            'Document all changes made',
            'Schedule follow-up audit'
        ]

        for item in checklist_items:
            sheet[f'A{row}'] = '☐'
            sheet[f'B{row}'] = item
            sheet[f'B{row}'].font = Font(name='Calibri', size=11)
            row += 1

        self.auto_adjust_columns(sheet)

    def auto_adjust_columns(self, sheet):
        """Auto-adjust column widths"""
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            sheet.column_dimensions[column_letter].width = adjusted_width