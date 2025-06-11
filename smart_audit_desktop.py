import os
import csv
import PySimpleGUI as sg
from comprehensive_audit_engine import ComprehensiveAuditEngine

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

def run_audit(target_dir, files):
    config = {'enable_ai': True, 'zero_change_mode': True}
    engine = ComprehensiveAuditEngine(config)
    audit_results = engine.run_comprehensive_audit(target_dir)
    analyzed = audit_results.get('files_analyzed', [])
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

def main():
    sg.theme('LightBlue')
    layout = [
        [sg.Text('Smart Audit Desktop', font=('Inter', 20, 'bold'))],
        [sg.Text('Επιλογή φακέλου εργασίας:'), sg.Input(key='FOLDER', enable_events=True, readonly=True), sg.FolderBrowse('Browse')],
        [sg.Button('Σάρωση & Audit', key='SCAN'), sg.Button('Έξοδος')],
        [sg.Text('', key='MSG', size=(60,1), text_color='red')],
        [sg.Table(
            values=[],
            headings=['Αρχείο', 'Τρέχων φάκελος', 'Πρόταση συστήματος', 'Επιλογή φακέλου'],
            key='TABLE',
            auto_size_columns=False,
            col_widths=[30, 20, 30, 20],
            display_row_numbers=False,
            justification='left',
            enable_events=True,
            select_mode=sg.TABLE_SELECT_MODE_NONE,
            num_rows=15,
            visible=False
        )],
        [sg.Button('Export Επιλογών', key='EXPORT', visible=False)]
    ]
    window = sg.Window('Smart Audit Desktop', layout, finalize=True, resizable=True)

    files_data = []
    folders = []
    table_data = []
    drop_downs = []

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Έξοδος'):
            break
        if event == 'SCAN':
            folder = values['FOLDER']
            if not folder or not os.path.isdir(folder):
                window['MSG'].update('Ο φάκελος δεν υπάρχει!')
                continue
            window['MSG'].update('Σάρωση και ανάλυση...')
            files_data = scan_files(folder)
            folders = get_all_folders(folder)
            table_data = run_audit(folder, files_data)
            # Prepare table for display
            table_display = []
            drop_downs = []
            for i, f in enumerate(table_data):
                table_display.append([
                    f['filename'],
                    f['folder'],
                    f['system_suggestion'],
                    folders.index(f['folder']) if f['folder'] in folders else 0
                ])
                drop_downs.append(sg.Combo(folders, default_value=f['folder'], key=f'DROP_{i}', size=(20,1)))
            window['TABLE'].update(values=[[d[0], d[1], d[2], folders[d[3]]] for d in table_display], visible=True)
            window['EXPORT'].update(visible=True)
            window['MSG'].update(f'Βρέθηκαν {len(table_data)} αρχεία.')
        if event == 'EXPORT':
            # Read user selections from drop-downs
            for i, f in enumerate(table_data):
                sel = window[f'DROP_{i}'].get() if window[f'DROP_{i}'] else f['folder']
                table_data[i]['selected_folder'] = sel
            # Export to CSV
            save_path = sg.popup_get_file('Αποθήκευση επιλογών ως CSV', save_as=True, file_types=(('CSV Files', '*.csv'),), default_extension='csv')
            if save_path:
                with open(save_path, 'w', encoding='utf-8', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'full_path', 'selected_folder', 'system_suggestion'])
                    writer.writeheader()
                    for row in table_data:
                        writer.writerow({
                            'filename': row['filename'],
                            'full_path': row['full_path'],
                            'selected_folder': row['selected_folder'],
                            'system_suggestion': row['system_suggestion']
                        })
                sg.popup('Επιτυχής εξαγωγή!')
    window.close()

if __name__ == '__main__':
    main()