import os
import webbrowser
import subprocess
import sys

def is_business_file(filename):
    # Ignore system, git, log, cache, pyc, hidden files
    ignore_ext = ['.log', '.pyc', '.tmp', '.DS_Store']
    ignore_dirs = ['.git', '__pycache__', 'file_organizer_cache']
    if any(filename.endswith(ext) for ext in ignore_ext):
        return False
    if filename.startswith('.'):
        return False
    return True

def scan_files(root_dir):
    business_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip ignored directories
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
        # Skip ignored directories
        dirnames[:] = [d for d in dirnames if d not in ['.git', '__pycache__', 'file_organizer_cache']]
        for d in dirnames:
            folders.add(os.path.relpath(os.path.join(dirpath, d), root_dir))
    folders.add('.')
    return sorted(folders)

def generate_selector_html(files, folders, html_path):
    html = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Επιλογή φακέλου για αρχεία</title>',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">',
        '<style>',
        'body { font-family: "Inter", sans-serif; background: #f4f6f8; color: #1e293b; margin: 0; }',
        '.container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0002; padding: 32px; }',
        'h1 { font-size: 2rem; margin-bottom: 24px; }',
        'table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }',
        'th, td { padding: 14px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }',
        'th { background: #f8fafc; font-weight: 600; color: #475569; }',
        'tr:last-child td { border-bottom: none; }',
        'select { width: 100%; padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; color: #1e293b; font-size: 0.95rem; }',
        'select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 2px #3b82f633; }',
        '.export-btn { margin-top: 24px; padding: 12px 24px; font-size: 1rem; font-weight: 500; color: #fff; background: #3b82f6; border: none; border-radius: 6px; cursor: pointer; transition: all 0.2s; }',
        '.export-btn:hover { background: #2563eb; }',
        '</style>',
        '</head>',
        '<body>',
        '<div class="container">',
        '<h1>Επιλογή φακέλου για κάθε αρχείο</h1>',
        '<table>',
        '<tr><th>Αρχείο</th><th>Τρέχων φάκελος</th><th>Επιλογή φακέλου</th></tr>'
    ]
    for i, f in enumerate(files):
        html.append(f'<tr><td>{f["filename"]}</td><td>{f["folder"]}</td><td><select id="sel_{i}">')
        for folder in folders:
            selected = 'selected' if folder == f['folder'] else ''
            html.append(f'<option value="{folder}" {selected}>{folder if folder != "." else "(root)"}</option>')
        html.append('</select></td></tr>')
    html.append('</table>')
    html.append('<button class="export-btn" onclick="exportChoices()">Export Choices & Start Audit</button>')
    html.append('</div>')
    html.append('<script>')
    html.append(f'const files = {files!r};')
    html.append('''
function exportChoices() {
  let csv = 'filename,full_path,selected_folder\n';
  files.forEach((f, i) => {
    let sel = document.getElementById('sel_' + i);
    let folderVal = sel ? sel.value : '';
    csv += '"'+f.filename+'","'+f.full_path+'","'+folderVal+'"\n';
  });
  let blob = new Blob([csv], {type:'text/csv'});
  let a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'folder_choices.csv';
  a.click();
  // Μετά το export, ξεκινά το audit script
  setTimeout(() => { fetch('http://localhost:8765/start_audit'); }, 1000);
}
''')
    html.append('</script>')
    html.append('</body></html>')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

def start_local_server_and_wait_for_audit(root_dir):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/start_audit':
                # Τρέξε το audit script με το root_dir
                subprocess.Popen([sys.executable, 'audit_html_report.py', root_dir])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Audit started')
            else:
                self.send_response(404)
                self.end_headers()
    server = HTTPServer(('localhost', 8765), Handler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    return server

def main():
    if len(sys.argv) != 2:
        print('Usage: python3 folder_selector.py <target_directory>')
        sys.exit(1)
    root_dir = sys.argv[1].strip()
    if not os.path.isdir(root_dir):
        print(f'Error: Directory "{root_dir}" does not exist')
        sys.exit(1)
    files = scan_files(root_dir)
    folders = get_all_folders(root_dir)
    html_path = os.path.join(root_dir, 'folder_selector.html')
    generate_selector_html(files, folders, html_path)
    print(f'HTML selector saved: {html_path}')
    # Start local server to listen for audit trigger
    server = start_local_server_and_wait_for_audit(root_dir)
    webbrowser.open('file://' + os.path.abspath(html_path))
    print('Waiting for you to export choices and trigger audit...')
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting.')
        server.shutdown()

if __name__ == '__main__':
    main()