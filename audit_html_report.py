import os
import webbrowser
import sys
from collections import defaultdict
from comprehensive_audit_engine import ComprehensiveAuditEngine

def get_all_folders(root_dir):
    """Return a sorted list of all folders (relative paths) under root_dir."""
    folders = set()
    for dirpath, dirnames, _ in os.walk(root_dir):
        for d in dirnames:
            folders.add(os.path.relpath(os.path.join(dirpath, d), root_dir))
    folders.add('.')  # Add root itself
    return sorted(folders)

def group_files_by_folder(files, root_dir):
    grouped = defaultdict(list)
    for file_info in files:
        folder = os.path.relpath(os.path.dirname(file_info.get('full_path', '')), root_dir)
        grouped[folder].append(file_info)
    return grouped

def generate_html_report(audit_results, html_path, root_dir):
    files = audit_results.get('files_analyzed', [])
    # Filter only files that need manual classification
    files_to_show = [
        f for f in files if (
            f.get('Requires_User_Decision') or
            f.get('Document_Type') in ['Unclassified', 'REVIEW_NEEDED', 'REVIEW_CLASSIFY']
        )
    ]
    all_folders = get_all_folders(root_dir)
    grouped = group_files_by_folder(files_to_show, root_dir)
    html = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Audit Results</title>',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">',
        '<style>',
        '* { margin: 0; padding: 0; box-sizing: border-box; }',
        'body { font-family: "Inter", sans-serif; background: #f4f6f8; color: #1e293b; }',
        '.container { display: flex; min-height: 100vh; }',
        '.sidebar { width: 280px; background: #fff; border-right: 1px solid #e2e8f0; position: fixed; height: 100vh; overflow-y: auto; }',
        '.sidebar-header { padding: 24px; border-bottom: 1px solid #e2e8f0; }',
        '.sidebar-header h2 { font-size: 1.25rem; font-weight: 600; color: #1e293b; }',
        '.folder-list { list-style: none; }',
        '.folder-list li { padding: 12px 24px; cursor: pointer; transition: all 0.2s; border-left: 3px solid transparent; }',
        '.folder-list li:hover { background: #f8fafc; color: #3b82f6; }',
        '.folder-list li.active { background: #f1f5f9; color: #3b82f6; border-left-color: #3b82f6; }',
        '.main { flex: 1; margin-left: 280px; padding: 32px; }',
        '.main-header { margin-bottom: 24px; }',
        '.main-header h2 { font-size: 1.5rem; font-weight: 600; color: #1e293b; }',
        '.main-header h2 span { color: #3b82f6; }',
        '.file-table { width: 100%; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }',
        '.file-table th, .file-table td { padding: 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }',
        '.file-table th { background: #f8fafc; font-weight: 600; color: #475569; }',
        '.file-table tr:last-child td { border-bottom: none; }',
        '.file-table select { width: 100%; padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; color: #1e293b; font-size: 0.875rem; }',
        '.file-table select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.1); }',
        '.export-btn { margin-top: 24px; padding: 12px 24px; font-size: 0.875rem; font-weight: 500; color: #fff; background: #3b82f6; border: none; border-radius: 6px; cursor: pointer; transition: all 0.2s; }',
        '.export-btn:hover { background: #2563eb; }',
        '.empty-state { text-align: center; padding: 48px; color: #64748b; }',
        '@media (max-width: 768px) {',
        '  .container { flex-direction: column; }',
        '  .sidebar { width: 100%; height: auto; position: relative; }',
        '  .main { margin-left: 0; padding: 16px; }',
        '}',
        '</style>',
        '</head>',
        '<body>',
        '<div class="container">',
        '<div class="sidebar">',
        '<div class="sidebar-header">',
        '<h2>📁 Folders</h2>',
        '</div>',
        '<ul class="folder-list" id="folderList">'
    ]
    for idx, folder in enumerate(all_folders):
        display = folder if folder != '.' else '(root)'
        html.append(f'<li data-folder="{folder}" onclick="showFolder(\'{folder}\')" id="folder-li-{idx}">{display}</li>')
    html.append('</ul></div>')
    html.append('<div class="main" id="mainContent">')
    html.append('<div class="main-header"><h2>Επιλέξτε φάκελο από αριστερά</h2></div>')
    html.append('</div></div>')
    # JavaScript for dynamic folder/file display
    html.append('<script>')
    # Data for JS
    html.append('const folderData = {};')
    for folder in all_folders:
        files = grouped.get(folder, [])
        html.append(f'folderData["{folder}"] = [')
        for file_info in files:
            filename = file_info.get('filename', '').replace('"', '\"')
            doc_type = file_info.get('document_type', '').replace('"', '\"')
            explanation = file_info.get('explanation', '').replace('"', '\"')
            full_path = file_info.get('full_path', '').replace('"', '\"')
            html.append('{')
            html.append(f'filename: "{filename}", doc_type: "{doc_type}", explanation: "{explanation}", full_path: "{full_path}"')
            html.append('},')
        html.append('];')
    # All folders for drop-downs
    html.append('const allFolders = [')
    for folder in all_folders:
        html.append(f'"{folder}",')
    html.append('];')
    # JS functions
    html.append('''
function showFolder(folder) {
  // Highlight selected
  document.querySelectorAll('.folder-list li').forEach(li => li.classList.remove('active'));
  let idx = allFolders.indexOf(folder);
  if (idx >= 0) document.getElementById('folder-li-' + idx).classList.add('active');
  // Show files
  let files = folderData[folder] || [];
  let html = `<div class=\"main-header\"><h2>Αρχεία προς χειροκίνητη ταξινόμηση: <span>${folder === '.' ? '(root)' : folder}</span></h2></div>`;
  if (files.length === 0) {
    html += '<div class="empty-state"><p>Δεν υπάρχουν αρχεία προς χειροκίνητη ταξινόμηση σε αυτόν τον φάκελο.</p></div>';
  } else {
    html += `<table class=\"file-table\"><tr><th>Filename</th><th>Document Type</th><th>Move to Folder</th><th>Explanation</th></tr>`;
    files.forEach((f, i) => {
      html += `<tr><td>${f.filename}</td><td>${f.doc_type}</td><td><select id=\"sel_${i}\">`;
      allFolders.forEach(opt => {
        html += `<option value=\"${opt}\">${opt === '.' ? '(root)' : opt}</option>`;
      });
      html += `</select></td><td>${f.explanation}</td></tr>`;
    });
    html += '</table>';
    html += `<button class=\"export-btn\" onclick=\"exportChoices('${folder}')\">Export Choices to CSV</button>`;
  }
  document.getElementById('mainContent').innerHTML = html;
}
function exportChoices(folder) {
  let files = folderData[folder] || [];
  let csv = 'Filename,Document Type,Selected Folder\n';
  files.forEach((f, i) => {
    let sel = document.getElementById('sel_' + i);
    let folderVal = sel ? sel.value : '';
    csv += '"'+f.filename+'","'+f.doc_type+'","'+folderVal+'"\n';
  });
  let blob = new Blob([csv], {type:'text/csv'});
  let a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'audit_choices_' + folder.replace(/\\W/g,'_') + '.csv';
  a.click();
}
// Auto-select first folder
window.onload = function() { showFolder(allFolders[0]); };
''')
    html.append('</script>')
    html.append('</body></html>')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

def main():
    if len(sys.argv) != 2:
        print('Usage: python3 audit_html_report.py <target_directory>')
        sys.exit(1)

    target_dir = sys.argv[1].strip()
    if not os.path.isdir(target_dir):
        print(f'Error: Directory "{target_dir}" does not exist')
        sys.exit(1)

    print(f'Running audit on directory: {target_dir}')
    config = {'enable_ai': True, 'zero_change_mode': True}
    engine = ComprehensiveAuditEngine(config)
    print('Running audit...')
    audit_results = engine.run_comprehensive_audit(target_dir)
    html_path = os.path.join(target_dir, 'audit_results.html')
    generate_html_report(audit_results, html_path, target_dir)
    abs_html_path = os.path.abspath(html_path)
    print(f'HTML report saved: {abs_html_path}')
    webbrowser.open('file://' + abs_html_path)

if __name__ == '__main__':
    main()