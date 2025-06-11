# 🔍 COMPREHENSIVE FILE AUDIT SYSTEM v2.0

## Επισκόπηση

Το **Comprehensive File Audit System** είναι μια εκτεταμένη λύση ελέγχου αρχείων που ενσωματώνει AI και παρέχει μια ολοκληρωμένη ανάλυση της δομής των αρχείων σας. Το σύστημα ακολουθεί τη φιλοσοφία **"zero-change"** - κάνει μόνο έλεγχο και παρέχει συστάσεις, χωρίς να αλλάζει τίποτα αυτόματα.

## 🎯 Χαρακτηριστικά

### ✅ Υλοποιημένοι Έλεγχοι (Βάση των 13 Απαιτήσεων)

1. **🗂️ Ανίχνευση άσχετων/μη κατηγοριοποιημένων φακέλων**
   - Εντοπίζει φακέλους όπως: `Argyrhs`, `Anna`, `TEMPORARY`, `DELETED`
   - Παρέχει dropdown επιλογές για δράση

2. **🔍 Ανίχνευση διπλών αρχείων**
   - Content-based detection (MD5 hash)
   - Fuzzy name matching για παρόμοια ονόματα
   - Σύγκριση μεγεθών και ημερομηνιών

3. **📝 Ανίχνευση κακών ονομάτων αρχείων**
   - Patterns όπως: `Container 1`, `Copy`, `Doc1`, `Version_2`
   - Προειδοποιήσεις για μετονομασία

4. **🚢 Καταγραφή ημερομηνιών φόρτωσης/άφιξης**
   - OCR parsing σε HBL, BL, TLX αρχεία
   - Εντοπισμός `Date of Loading` και `Date of Arrival`

5. **💰 Υπολογισμός τελικού κόστους τελωνείου**
   - OCR/text search για κόστη τελωνείου
   - Λέξεις-κλειδιά: "Τελωνείο", "Δασμοί", "Σύνολο προς πληρωμή"

6. **📋 Ανίχνευση orphan CE**
   - Σύγκριση CE certificates με προϊόντα από invoices
   - Εντοπισμός CE που δεν ταιριάζουν με προϊόντα

7. **🔗 Ανίχνευση shared CE**
   - CE που εμφανίζονται σε πολλά containers
   - Grouped CE analysis

8. **📊 Ανίχνευση ελλειπόντων εγγράφων**
   - Containers με Invoice & CE αλλά χωρίς Packing List
   - Completeness scoring

9. **👨‍👩‍👧‍👦 Ανίχνευση αρχείων για γονείς**
   - Patterns: "πατέρας", "στείλε στους", "να δει ο μπαμπάς"
   - Ξεχωριστή κατηγορία

10. **🖥️ Interactive UI**
    - Progress bars και χρονόμετρα
    - Excel dropdown menus
    - Live statistics

11. **🛡️ Zero-change philosophy**
    - Κανένας αυτόματος μετακίνηση/αλλαγή
    - Μόνο recommendations

12. **🤖 AI Integration**
    - Llama 3.2:3b για consistency checking
    - OCR με Tesseract
    - Fuzzy matching προϊόντων

13. **📈 Comprehensive Excel Output**
    - Multiple sheets με dropdown menus
    - Conditional formatting
    - Interactive charts

## 📦 Εγκατάσταση

### Προαπαιτούμενα

```bash
# Python 3.8+
# Εξαρτήσεις εγκαθίστανται αυτόματα
pip install pandas openpyxl pytesseract PyMuPDF transformers torch sentence-transformers scikit-learn psutil rich
```

### Llama Model (Προαιρετικό για AI features)

```bash
# Εγκατάσταση Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download Llama model
ollama pull llama3.2:3b
```

## 🚀 Χρήση

### Βασική Χρήση

```bash
# Έλεγχος φακέλου με όλα τα features
python3 run_comprehensive_audit.py --target "/path/to/business/documents"

# Με custom output file
python3 run_comprehensive_audit.py --target "/path/to/docs" --output "my_audit.xlsx"

# Χωρίς AI για ταχύτερη εκτέλεση
python3 run_comprehensive_audit.py --target "/path/to/docs" --no-ai

# Verbose logging για debugging
python3 run_comprehensive_audit.py --target "/path/to/docs" --verbose

# Dry run για testing
python3 run_comprehensive_audit.py --target "/path/to/docs" --dry-run
```

### Με Configuration File

```bash
# Χρήση custom config
python3 run_comprehensive_audit.py --target "/path/to/docs" --config audit_config.json
```

## ⚙️ Διαμόρφωση

### audit_config.json

```json
{
    "log_level": "INFO",
    "enable_ai": true,
    "enable_ocr": true,
    "similarity_threshold": 0.85,
    "problematic_folders": [
        "argyrhs", "anna", "temporary", "deleted"
    ],
    "known_suppliers": ["queena", "argy", "anna", "frankouli"],
    "required_documents": ["invoice", "ce", "packing_list"],
    "ai_models": {
        "llama_model": "llama3.2:3b",
        "llama_timeout": 60
    }
}
```

## 📊 Excel Report Structure

Το output Excel file περιέχει τα εξής sheets:

### 📊 Σύνοψη Audit
- Συνολικά στατιστικά
- Breakdown προβλημάτων
- Performance metrics

### 📁 Ανάλυση Αρχείων
Βασισμένο στη δομή που ζητήθηκε:
```
Supplier | Year | Container | Invoice | PI | CE % | CE Missing List |
Packing List | BL Found | Load Date | Arrival Date | Customs Cost |
Orphans | Shared CE | Duplicate Files | Unknown Folders | Notes | Status
```

**Dropdown Menus:**
- Status: `COMPLETE`, `INCOMPLETE`, `NEEDS_REVIEW`, `IN_PROGRESS`
- Recommended Action: `KEEP`, `DELETE`, `REVIEW`, `MOVE`, `RENAME`

### ⚠️ Προβλήματα
- File path και περιγραφή προβλήματος
- Severity levels
- Recommended actions με dropdowns

### 🗂️ Άγνωστοι Φάκελοι
- Φάκελοι που δεν ταιριάζουν σε patterns
- Προτάσεις για αντιμετώπιση
- Action dropdowns

### 🔍 Διπλά Αρχεία
- Content duplicates
- Name similarities
- File details και προτάσεις

### 📋 Πληρότητα
- Document completeness per container
- Missing document analysis
- Completeness scores

### 💡 Προτάσεις
- Actionable recommendations
- Priority levels
- Implementation checklist

## 🧪 Testing

```bash
# Απλό test χωρίς AI/OCR
python3 simple_audit_test.py

# Πλήρες test με AI features
python3 test_audit_system.py
```

## 📈 Output Παράδειγμα

```
✅ AUDIT COMPLETED SUCCESSFULLY
================================================================================
📊 Total Files Analyzed: 349
⚠️  Issues Found: 23
🤖 AI Success Rate: 67.2%
⏱️  Processing Time: 45.3 seconds
📈 Files per Second: 7.7

📄 Comprehensive report saved: comprehensive_audit_report.xlsx
📂 Log files saved in: ./audit_logs/

💡 NEXT STEPS:
   1. Review the Excel report with all audit findings
   2. Use dropdown menus in Excel for action selection
   3. Start with HIGH PRIORITY items first
   4. Remember: This audit made NO changes to your files
   5. Create backups before implementing any recommendations
```

## 🔧 Αρχιτεκτονική

### Core Modules

- **`advanced_audit_system.py`**: Κεντρικό audit engine
- **`excel_report_generator.py`**: Excel reporting με styling
- **`run_comprehensive_audit.py`**: CLI interface
- **Integration με `super_file_organizer.py`**: AI και pattern classification

### Extensibility

Το σύστημα είναι σχεδιασμένο για επεκτασιμότητα:

```python
# Προσθήκη νέου ελέγχου
class CustomAuditChecker:
    def check_custom_rule(self, file_info):
        # Custom logic
        return result

# Ενσωμάτωση στο engine
audit_engine.add_custom_checker(CustomAuditChecker())
```

## 📋 Roadmap

### Επόμενες βελτιώσεις:
- [ ] Streamlit web interface
- [ ] Real-time monitoring
- [ ] Custom rule engine
- [ ] API endpoints
- [ ] Cloud storage integration
- [ ] Automated scheduling

## 🐛 Troubleshooting

### Συνήθη προβλήματα:

**OCR Errors:**
```bash
# Εγκατάσταση Tesseract
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

**Llama Timeouts:**
```json
{
    "ai_models": {
        "llama_timeout": 120  // Αύξηση timeout
    }
}
```

**Memory Issues:**
```json
{
    "max_threads": 2,  // Μείωση threads
    "enable_ai": false  // Απενεργοποίηση AI
}
```

## 📞 Support

Για προβλήματα ή questions:
1. Ελέγξτε τα log files στο `./audit_logs/`
2. Τρέξτε με `--verbose` για περισσότερες πληροφορίες
3. Χρησιμοποιήστε `--dry-run` για testing

---

**Created with ❤️ for enterprise file organization**