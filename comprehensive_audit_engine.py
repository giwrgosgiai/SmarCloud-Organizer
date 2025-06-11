import os
import re
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Set

class ComprehensiveAuditEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cancelled = False
        self.progress_callback = None
        self.folder_cache = {}  # Cache for folder contents
        self.current_folder = None
        self.current_folder_contents = None

    def run_comprehensive_audit(self, target_dir: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a comprehensive audit on the target directory.

        Args:
            target_dir: The directory to audit
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary containing audit results
        """
        self.progress_callback = progress_callback
        self.cancelled = False

        results = {
            'timestamp': datetime.now().isoformat(),
            'target_directory': target_dir,
            'files_analyzed': [],
            'issues_found': [],
            'summary': {}
        }

        # Get all files in directory
        all_files = []
        for root, _, files in os.walk(target_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        total_files = len(all_files)

        # Analyze each file
        for idx, file_path in enumerate(all_files, 1):
            if self.cancelled:
                break

            if self.progress_callback:
                self.progress_callback(idx, total_files, f"Analyzing {os.path.basename(file_path)}")

            file_info = self._analyze_file(file_path)
            if file_info:
                results['files_analyzed'].append(file_info)

                # Check for issues
                issues = self._check_issues(file_path, file_info)
                if issues:
                    results['issues_found'].extend(issues)

        # Generate summary
        results['summary'] = self._generate_summary(results)

        return results

    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file"""
        try:
            # Get file info
            stat = os.stat(file_path)
            file_info = {
                'filename': os.path.basename(file_path),
                'full_path': file_path,
                'size': stat.st_size,
                'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'document_type': None,
                'issues': []
            }

            # Detect document type
            file_info['document_type'] = self._detect_document_type(file_path)

            # Check for issues
            issues = self._check_issues(file_path, file_info)
            file_info['issues'] = issues

            # Check folder consistency
            folder_issues = self._check_folder_consistency(file_path, file_info)
            file_info['issues'].extend(folder_issues)

            return file_info

        except Exception as e:
            return {
                'filename': os.path.basename(file_path),
                'full_path': file_path,
                'error': str(e),
                'document_type': 'unknown',
                'modified_date': 'unknown',
                'issues': [{'type': 'analysis_error', 'severity': 'error', 'message': str(e)}]
            }

    def _detect_document_type(self, file_path: str) -> str:
        """Detect the type of document based on filename patterns"""
        filename = os.path.basename(file_path)
        filename_lower = filename.lower()

        # Common patterns for different document types
        patterns = {
            'invoice': r'invoice|inv|bill',
            'packing_list': r'packing|packing list|pl',
            'bill_of_lading': r'bl|bill of lading|b/l',
            'certificate': r'certificate|cert|ce',
            'manual': r'manual|guide|instructions',
            'price_list': r'price|pricelist|pricing'
        }

        for doc_type, pattern in patterns.items():
            if re.search(pattern, filename_lower):
                return doc_type

        return 'unknown'

    def _check_issues(self, file_path: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for potential issues with the file"""
        issues = []

        # Check for missing document type
        if file_info['document_type'] == 'unknown':
            issues.append({
                'type': 'unknown_document_type',
                'severity': 'warning',
                'message': f"Could not determine document type for {file_info['filename']}",
                'file_info': file_info
            })

        # Check for very old files
        mod_date = file_info.get('modified_date', 'unknown')
        if mod_date and mod_date != 'unknown':
            try:
                modified_date = datetime.fromisoformat(mod_date)
                if (datetime.now() - modified_date).days > 365:
                    issues.append({
                        'type': 'old_file',
                        'severity': 'info',
                        'message': f"File {file_info['filename']} is more than 1 year old",
                        'file_info': file_info
                    })
            except Exception:
                pass  # Ignore invalid date

        # Check folder consistency
        folder_issues = self._check_folder_consistency(file_path, file_info)
        issues.extend(folder_issues)

        return issues

    def _check_folder_consistency(self, file_path: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if file is in the correct folder based on its content"""
        issues = []
        current_folder = os.path.dirname(file_path)

        # If we're in a new folder, update the cache
        if current_folder != self.current_folder:
            self.current_folder = current_folder
            self.current_folder_contents = self._get_folder_contents(current_folder)

        # Extract container numbers and other identifiers
        container_numbers = self._extract_container_numbers(file_info)
        hbl_numbers = self._extract_hbl_numbers(file_info)

        # Check if file should be in a container folder
        if container_numbers:
            for container in container_numbers:
                container_folder = self._find_container_folder(container)
                if container_folder and container_folder != current_folder:
                    issues.append({
                        'type': 'folder_consistency',
                        'severity': 'warning',
                        'message': f'File mentions Container {container} but is in {os.path.basename(current_folder)}',
                        'suggestion': f'Move to {os.path.basename(container_folder)} folder',
                        'file_info': file_info
                    })

        # Check if file should be in an HBL folder
        if hbl_numbers:
            for hbl in hbl_numbers:
                hbl_folder = self._find_hbl_folder(hbl)
                if hbl_folder and hbl_folder != current_folder:
                    issues.append({
                        'type': 'folder_consistency',
                        'severity': 'warning',
                        'message': f'File is related to HBL {hbl} but is in {os.path.basename(current_folder)}',
                        'suggestion': f'Move to {os.path.basename(hbl_folder)} folder',
                        'file_info': file_info
                    })

        # Check if invoice should be in Invoice folder
        if file_info['document_type'] == 'Invoice':
            invoice_folder = self._find_invoice_folder(current_folder)
            if invoice_folder and invoice_folder != current_folder:
                issues.append({
                    'type': 'folder_consistency',
                    'severity': 'warning',
                    'message': 'Invoice should be in Invoice folder',
                    'suggestion': f'Move to {os.path.basename(invoice_folder)} folder',
                    'file_info': file_info
                })

        return issues

    def _get_folder_contents(self, folder_path: str) -> Dict[str, Any]:
        """Get and cache folder contents"""
        if folder_path in self.folder_cache:
            return self.folder_cache[folder_path]

        contents = {
            'files': [],
            'subfolders': [],
            'container_numbers': set(),
            'hbl_numbers': set()
        }

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                contents['files'].append(file_path)

                # Extract identifiers from filename
                container_nums = self._extract_container_numbers_from_filename(file)
                hbl_nums = self._extract_hbl_numbers_from_filename(file)

                contents['container_numbers'].update(container_nums)
                contents['hbl_numbers'].update(hbl_nums)

            for dir_name in dirs:
                contents['subfolders'].append(os.path.join(root, dir_name))

        self.folder_cache[folder_path] = contents
        return contents

    def _extract_container_numbers_from_filename(self, filename: str) -> Set[str]:
        """Extract container numbers from filename"""
        container_numbers = set()

        # Look for patterns like TCNU4982977/40
        container_pattern = r'([A-Z]{4}\d{7}/\d{2})'
        matches = re.finditer(container_pattern, filename)
        container_numbers.update(match.group(1) for match in matches)

        return container_numbers

    def _extract_hbl_numbers_from_filename(self, filename: str) -> Set[str]:
        """Extract HBL numbers from filename"""
        hbl_numbers = set()

        # Look for patterns like HBLNOACAXMN24080011
        hbl_pattern = r'(HBL[A-Z0-9]{8,})'
        matches = re.finditer(hbl_pattern, filename)
        hbl_numbers.update(match.group(1) for match in matches)

        return hbl_numbers

    def _find_container_folder(self, container_number: str) -> Optional[str]:
        """Find the appropriate container folder"""
        parent_folder = os.path.dirname(self.current_folder)

        # Look for folder containing the container number
        for root, dirs, _ in os.walk(parent_folder):
            for dir_name in dirs:
                if container_number in dir_name:
                    return os.path.join(root, dir_name)

        return None

    def _find_hbl_folder(self, hbl_number: str) -> Optional[str]:
        """Find the appropriate HBL folder"""
        parent_folder = os.path.dirname(self.current_folder)

        # Look for folder containing the HBL number
        for root, dirs, _ in os.walk(parent_folder):
            for dir_name in dirs:
                if hbl_number in dir_name:
                    return os.path.join(root, dir_name)

        return None

    def _find_invoice_folder(self, current_folder: str) -> Optional[str]:
        """Find the Invoice folder within the current folder's parent"""
        parent_folder = os.path.dirname(current_folder)
        invoice_folder = os.path.join(parent_folder, 'Invoice')

        if os.path.exists(invoice_folder):
            return invoice_folder

        return None

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the audit results"""
        summary = {
            'total_files': len(results['files_analyzed']),
            'total_issues': len(results['issues_found']),
            'document_types': {},
            'issue_types': {}
        }

        # Count document types
        for file_info in results['files_analyzed']:
            doc_type = file_info.get('document_type', 'unknown')
            summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1

        # Count issue types
        for issue in results['issues_found']:
            issue_type = issue.get('type', 'unknown')
            summary['issue_types'][issue_type] = summary['issue_types'].get(issue_type, 0) + 1

        return summary

    def cancel(self):
        """Cancel the current audit"""
        self.cancelled = True

    def _extract_container_numbers(self, file_info: dict) -> set:
        """Extract container numbers from file_info or filename."""
        filename = file_info.get('filename', '')
        pattern = r'([A-Z]{4}\d{7}/\d{2})'
        return set(re.findall(pattern, filename))

    def _extract_hbl_numbers(self, file_info: dict) -> set:
        """Extract HBL numbers from file_info or filename."""
        filename = file_info.get('filename', '')
        pattern = r'(HBL[A-Z0-9]{8,})'
        return set(re.findall(pattern, filename))