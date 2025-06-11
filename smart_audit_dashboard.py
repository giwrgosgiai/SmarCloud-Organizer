import os
import sys
from flask import Flask, render_template_string, request, redirect, url_for, send_file, jsonify
import threading
import tempfile
import csv
import webbrowser
from comprehensive_audit_engine import ComprehensiveAuditEngine

app = Flask(__name__)

# --- HTML Templates ---
TEMPLATE_INDEX = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Smart Audit Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #f4f6f8; color: #1e293b; margin: 0; }
        .container { max-width: 1100px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0002; padding: 32px; }
        h1 { font-size: 2rem; margin-bottom: 24px; }
        label { font-weight: 600; }
        input[type=text] { width: 70%; padding: 8px; border-radius: 6px; border: 1px solid #e2e8f0; margin-right: 12px; }
        button { padding: 10px 22px; border-radius: 6px; border: none; background: #3b82f6; color: #fff; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        .file-list { margin-top: 32px; }
        .file-list table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
        .file-list th, .file-list td { padding: 14px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }
        .file-list th { background: #f8fafc; font-weight: 600; color: #475569; }
        .file-list tr:last-child td { border-bottom: none; }
        .file-list select { width: 100%; padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; color: #1e293b; font-size: 0.95rem; }
        .file-list select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 2px #3b82f633; }
        .actions { margin-top: 32px; }
        .export-btn { margin-right: 16px; }
        .success { color: #16a34a; font-weight: 600; }
        .error { color: #dc2626; font-weight: 600; }
    </style>
</head>
<body>
<div class="container">
    <h1>Smart Audit Dashboard</h1>
    <form id="folderForm" method="post" action="/scan" enctype="multipart/form-data">
        <label for="folderPicker">Επιλογή φακέλου εργασίας:</label>
        <input type="file" id="folderPicker" name="folderPicker" webkitdirectory directory multiple required>
        <input type="hidden" id="folder" name="folder" value="">
        <button type="submit">Σάρωση & Audit</button>
    </form>
    <script>
    document.getElementById('folderForm').onsubmit = function(e) {
        var files = document.getElementById('folderPicker').files;
        if (files.length === 0) {
            alert('Παρακαλώ επιλέξτε φάκελο!');
            e.preventDefault();
            return false;
        }
        // Get the root folder from the first file's webkitRelativePath
        var firstPath = files[0].webkitRelativePath;
        var rootFolder = firstPath.split('/')[0];
        document.getElementById('folder').value = rootFolder;
        return true;
    };
    </script>
    {% if files %}
    <form method="post" action="/assign">
        <div class="file-list">
            <table>
                <tr><th>Αρχείο</th><th>Τρέχων φάκελος</th><th>Πρόταση συστήματος</th><th>Επιλογή φακέλου</th></tr>
                {% for f in files %}
                <tr>
                    <td>{{ f['filename'] }}</td>
                    <td>{{ f['folder'] }}</td>
                    <td>{{ f['system_suggestion'] }}</td>
                    <td>
                        <select name="sel_{{ loop.index0 }}">
                        {% for folder in folders %}
                            <option value="{{ folder }}" {% if folder == f['selected_folder'] %}selected{% endif %}>{{ folder if folder != '.' else '(root)' }}</option>
                        {% endfor %}
                        </select>
                        <input type="hidden" name="file_{{ loop.index0 }}" value="{{ f['full_path'] }}">
                        <input type="hidden" name="name_{{ loop.index0 }}" value="{{ f['filename'] }}">
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <div class="actions">
            <button class="export-btn" type="submit">Export Επιλογών</button>
        </div>
    </form>
    {% endif %}
    {% if message %}<div class="success">{{ message }}</div>{% endif %}
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
</div>
</body>
</html>
'''

TEMPLATE_RESULTS = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Αποτελέσματα Audit</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #f4f6f8; color: #1e293b; margin: 0; }
        .container { max-width: 1100px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0002; padding: 32px; }
        h1 { font-size: 2rem; margin-bottom: 24px; }
        table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
        th, td { padding: 14px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }
        th { background: #f8fafc; font-weight: 600; color: #475569; }
        tr:last-child td { border-bottom: none; }
        .export-btn { margin-top: 24px; padding: 12px 24px; font-size: 1rem; font-weight: 500; color: #fff; background: #3b82f6; border: none; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
        .export-btn:hover { background: #2563eb; }
    </style>
</head>
<body>
<div class="container">
    <h1>Αποτελέσματα Audit</h1>
    <table>
        <tr>
            <th>Αρχείο</th><th>Τύπος Εγγράφου</th><th>Εξήγηση</th><th>Προτεινόμενη Ενέργεια</th>
        </tr>
        {% for f in results %}
        <tr>
            <td>{{ f.get('filename', f.get('File_Name', '')) }}</td>
            <td>{{ f.get('document_type', f.get('Document_Type', '')) }}</td>
            <td>{{ f.get('explanation', f.get('Notes', '')) }}</td>
            <td>{{ f.get('Recommended_Action', '') }}</td>
        </tr>
        {% endfor %}
    </table>
    <form method="get" action="/export_results">
        <button class="export-btn" type="submit">Export σε CSV</button>
    </form>
</div>
</body>
</html>
'''

# --- Core Logic ---
def is_business_file(filename):
    ignore_ext = ['.log', '.pyc', '.tmp', '.DS_Store']
    if any(filename.endswith(ext) for ext in ignore_ext):
        return False
    if filename.startswith('.'):
        return False
    return True

def scan_files(root_dir):
    business_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'file_organizer_cache']]
        for fname in filenames:
            if is_business_file(fname):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                business_files.append({'filename': fname, 'full_path': full_path, 'rel_path': rel_path, 'folder': os.path.relpath(dirpath, root_dir)})
    return business_files

def get_all_folders(root_dir):
    folders = set()
    for dirpath, dirnames, _ in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'file_organizer_cache']]
        for d in dirnames:
            folders.add(os.path.relpath(os.path.join(dirpath, d), root_dir))
    folders.add('.')
    return sorted(folders)

# --- Flask Routes ---
selected_folder = ''
files_cache = []
folders_cache = []
assignments_cache = []
audit_results_cache = []
results_csv_path = ''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(TEMPLATE_INDEX, folder=selected_folder, files=files_cache, folders=folders_cache, message=None, error=None)

@app.route('/scan', methods=['POST'])
def scan():
    global selected_folder, files_cache, folders_cache
    # Use the uploaded files to determine the root folder
    if 'folderPicker' not in request.files:
        return render_template_string(TEMPLATE_INDEX, folder='', files=[], folders=[], message=None, error='Παρακαλώ επιλέξτε φάκελο!')
    files = request.files.getlist('folderPicker')
    if not files:
        return render_template_string(TEMPLATE_INDEX, folder='', files=[], folders=[], message=None, error='Παρακαλώ επιλέξτε φάκελο!')
    # Get the root folder from the first file's filename
    first_path = files[0].filename
    root_folder = first_path.split('/')[0]
    abs_root = os.path.abspath(root_folder)
    if not os.path.isdir(abs_root):
        return render_template_string(TEMPLATE_INDEX, folder=abs_root, files=[], folders=[], message=None, error='Ο φάκελος δεν υπάρχει!')
    selected_folder = abs_root
    files_list = scan_files(selected_folder)
    folders = get_all_folders(selected_folder)
    files_with_suggestions = run_audit(selected_folder, files_list)
    files_cache.clear()
    files_cache.extend(files_with_suggestions)
    folders_cache.clear()
    folders_cache.extend(folders)
    return render_template_string(TEMPLATE_INDEX, folder=selected_folder, files=files_cache, folders=folders_cache, message=f'Βρέθηκαν {len(files_cache)} αρχεία.', error=None)

@app.route('/assign', methods=['POST'])
def assign():
    global assignments_cache
    assignments = []
    for i, f in enumerate(files_cache):
        folder = request.form.get(f'sel_{i}', f['selected_folder'])
        assignments.append({'filename': f['filename'], 'full_path': f['full_path'], 'selected_folder': folder, 'system_suggestion': f['system_suggestion']})
    assignments_cache = assignments
    # Export to CSV
    fd, csv_path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'full_path', 'selected_folder', 'system_suggestion'])
        writer.writeheader()
        for row in assignments:
            writer.writerow(row)
    return send_file(csv_path, as_attachment=True, download_name='final_folder_choices.csv')

@app.route('/results', methods=['GET'])
def results():
    return render_template_string(TEMPLATE_RESULTS, results=audit_results_cache)

@app.route('/export_results', methods=['GET'])
def export_results():
    # Export audit results to CSV
    fd, csv_path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as csvfile:
        if audit_results_cache:
            fieldnames = list(audit_results_cache[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in audit_results_cache:
                writer.writerow(row)
    return send_file(csv_path, as_attachment=True, download_name='audit_results.csv')

# --- Audit Logic ---
def run_audit(target_dir, files):
    config = {'enable_ai': True, 'zero_change_mode': True}
    engine = ComprehensiveAuditEngine(config)
    print('Running audit...')
    audit_results = engine.run_comprehensive_audit(target_dir)
    analyzed = audit_results.get('files_analyzed', [])
    # Map audit results to scanned files
    file_map = {f['full_path']: f for f in analyzed}
    result = []
    for f in files:
        audit_info = file_map.get(f['full_path'], {})
        system_suggestion = audit_info.get('Suggested_New_Path') or audit_info.get('document_type') or audit_info.get('Document_Type') or ''
        result.append({
            'filename': f['filename'],
            'full_path': f['full_path'],
            'folder': f['folder'],
            'system_suggestion': system_suggestion,
            'selected_folder': f['folder'],
        })
    return result

# --- Main Entrypoint ---
def open_browser():
    import time
    time.sleep(1)
    webbrowser.open('http://127.0.0.1:5000/')

def main():
    threading.Thread(target=open_browser).start()
    app.run(debug=False, port=5000)

if __name__ == '__main__':
    main()