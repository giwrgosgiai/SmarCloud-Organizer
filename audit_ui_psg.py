import PySimpleGUI as sg
import threading
import os
import shutil
from comprehensive_audit_engine import ComprehensiveAuditEngine
from excel_report_generator import ExcelReportGenerator

# Helper to run audit in a thread
class AuditThread(threading.Thread):
    def __init__(self, config, target_dir, output_file, window):
        super().__init__()
        self.config = config
        self.target_dir = target_dir
        self.output_file = output_file
        self.window = window
        self.audit_results = None
        self.error = None

    def run(self):
        try:
            engine = ComprehensiveAuditEngine(self.config)
            def progress_callback(current, total, message):
                percent = int((current / total) * 100) if total else 0
                self.window.write_event_value('-PROGRESS-', (percent, message))
            self.audit_results = engine.run_comprehensive_audit(self.target_dir, progress_callback)
            # Generate report
            report = ExcelReportGenerator()
            report.generate_comprehensive_report(self.audit_results, self.output_file)
            self.window.write_event_value('-DONE-', self.audit_results)
        except Exception as e:
            self.error = str(e)
            self.window.write_event_value('-ERROR-', self.error)

def get_folder_options(file_info):
    """Return all relevant folder options for a file (current, suggested, Invoice, Container, HBL, etc)"""
    options = set()
    current_folder = os.path.dirname(file_info.get('full_path', ''))
    options.add(current_folder)
    explanation = file_info.get('explanation', '')
    # Add suggested folder if exists
    if 'Move to' in explanation:
        parts = explanation.split('Move to')
        if len(parts) > 1:
            folder_name = parts[1].split('folder')[0].strip()
            parent = os.path.dirname(current_folder)
            suggested_folder = os.path.join(parent, folder_name)
            options.add(suggested_folder)
    # Add Invoice folder if relevant
    if file_info.get('document_type', '').lower() == 'invoice':
        parent = os.path.dirname(current_folder)
        invoice_folder = os.path.join(parent, 'Invoice')
        options.add(invoice_folder)
    # Add container/HBL folders if found in explanation
    for word in explanation.split():
        if word.startswith('Container') or word.startswith('HBL'):
            parent = os.path.dirname(current_folder)
            folder = os.path.join(parent, word)
            options.add(folder)
    return list(options)

def move_file(source_path, dest_folder):
    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest_folder, filename)
    os.makedirs(dest_folder, exist_ok=True)
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(filename)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(dest_folder, f"{base}_{timestamp}{ext}")
    shutil.move(source_path, dest_path)
    return dest_path

def main():
    sg.theme('SystemDefault')
    layout = [
        [sg.Text('Target Directory'), sg.InputText(key='-TARGET-', size=(60,1)), sg.FolderBrowse()],
        [sg.Text('Output File'), sg.InputText('comprehensive_audit_report.xlsx', key='-OUTPUT-', size=(60,1)), sg.FileSaveAs(file_types=(('Excel Files', '*.xlsx'),))],
        [sg.Checkbox('Use AI Features', key='-AI-', default=True), sg.Checkbox('Zero-Change Mode', key='-ZERO-', default=True)],
        [sg.ProgressBar(100, orientation='h', size=(40, 20), key='-PROG-'), sg.Text('', key='-PROGMSG-', size=(40,1))],
        [sg.Button('Start Audit'), sg.Button('Cancel'), sg.Exit()],
        [sg.Text('Results:', font=('Any', 12, 'bold'))],
        [sg.Column([[]], key='-RESULTS-COL-', scrollable=True, vertical_scroll_only=True, size=(900, 300))]
    ]
    window = sg.Window('Comprehensive Audit System (PySimpleGUI)', layout, finalize=True, resizable=True)
    audit_thread = None
    config = {'enable_ai': True, 'zero_change_mode': True}
    file_widgets = []
    audit_results = None
    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Start Audit' and not audit_thread:
            target = values['-TARGET-']
            output = values['-OUTPUT-']
            config['enable_ai'] = values['-AI-']
            config['zero_change_mode'] = values['-ZERO-']
            if not target or not os.path.isdir(target):
                sg.popup_error('Please select a valid target directory!')
                continue
            window['-RESULTS-COL-'].update([[]])
            window['-PROG-'].update(0)
            window['-PROGMSG-'].update('Starting audit...')
            audit_thread = AuditThread(config, target, output, window)
            audit_thread.start()
        if event == 'Cancel' and audit_thread:
            sg.popup('Cancel not implemented in this demo.')
        if event == '-PROGRESS-':
            percent, msg = values['-PROGRESS-']
            window['-PROG-'].update(percent)
            window['-PROGMSG-'].update(msg)
        if event == '-DONE-':
            audit_results = values['-DONE-']
            files = audit_results.get('files_analyzed', [])
            file_widgets = []
            results_layout = []
            for idx, file_info in enumerate(files):
                folder_options = get_folder_options(file_info)
                default_folder = folder_options[0] if folder_options else os.path.dirname(file_info.get('full_path', ''))
                combo_key = f'-FOLDER-{idx}-'
                apply_key = f'-APPLY-{idx}-'
                results_layout.append([
                    sg.Text(file_info.get('filename', ''), size=(40,1)),
                    sg.Combo(folder_options, default_value=default_folder, key=combo_key, size=(50,1)),
                    sg.Button('Εφαρμογή', key=apply_key),
                    sg.Text(file_info.get('explanation', ''), size=(60,1), text_color='blue')
                ])
                file_widgets.append((file_info, combo_key, apply_key))
            if results_layout:
                results_layout.append([sg.Button('Εφαρμογή όλων', key='-APPLY-ALL-')])
            window['-RESULTS-COL-'].update(results_layout)
            window['-PROGMSG-'].update('Audit completed! Επιλέξτε φάκελο για κάθε αρχείο και πατήστε Εφαρμογή.')
            audit_thread = None
        if event and isinstance(event, str) and event.startswith('-APPLY-'):
            idx = int(event.split('-')[2])
            file_info, combo_key, _ = file_widgets[idx]
            target_folder = values[combo_key]
            try:
                dest_path = move_file(file_info['full_path'], target_folder)
                sg.popup(f'Το αρχείο μετακινήθηκε στο {dest_path}')
            except Exception as e:
                sg.popup_error(f'Αποτυχία μετακίνησης: {str(e)}')
        if event == '-APPLY-ALL-':
            errors = []
            for idx, (file_info, combo_key, _) in enumerate(file_widgets):
                target_folder = values[combo_key]
                try:
                    dest_path = move_file(file_info['full_path'], target_folder)
                except Exception as e:
                    errors.append(f"{file_info.get('filename','')}: {str(e)}")
            if errors:
                sg.popup_error('Κάποιες μετακινήσεις απέτυχαν:\n' + '\n'.join(errors))
            else:
                sg.popup('Όλα τα αρχεία μετακινήθηκαν επιτυχώς!')
    window.close()

if __name__ == '__main__':
    main()