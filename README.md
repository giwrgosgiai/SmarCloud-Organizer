# 🚀 Supercharged AI-Enhanced Zero-Loss File Organizer v3.0

**Το πιο προηγμένο AI-powered file organization system με υποστήριξη ελληνικών και zero-loss ασφάλεια!**

## ✨ Χαρακτηριστικά

### 🤖 **AI Features**
- **OCR Text Extraction** - Εξαγωγή κειμένου από PDF με υποστήριξη ελληνικών
- **Content-Based Classification** - AI ανάλυση περιεχομένου με transformers
- **Semantic Game Matching** - Έξυπνο matching αρχείων με games
- **Auto-Classification** - Αυτόματη κατηγοριοποίηση: CE, Manual, Invoice, Bank Proof, Packing List

### ⚡ **Advanced Features**
- **Zero-Loss Safety** - 100% ασφάλεια, χωρίς απώλεια αρχείων
- **Parallel Processing** - Γρήγορη επεξεργασία με πολλαπλά threads
- **Pattern & Fuzzy Matching** - Προηγμένο pattern recognition
- **Duplicate Detection** - Ανίχνευση διπλότυπων με hash comparison
- **Greek/English Support** - Πλήρης υποστήριξη ελληνικών
- **Excel Export** - Λεπτομερή reports σε Excel με multiple sheets

### 🛡️ **Safety Features**
- **Read-Only Analysis** - Δεν αλλάζει/μετακινεί αρχεία
- **Safety Verification** - Έλεγχος ότι δεν χάνονται αρχεία
- **Comprehensive Logging** - Πλήρη καταγραφή όλων των ενεργειών
- **Error Recovery** - Robust error handling

## 🔧 Εγκατάσταση

### 1. **Prerequisites**
```bash
# macOS (με Homebrew)
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ell

# Windows
# Download από: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. **Python Dependencies**
Το script κάνει **auto-install** όλα τα AI dependencies:
```bash
python3 super_file_organizer.py --install-deps
```

### 3. **Manual Installation** (αν χρειαστεί)
```bash
pip install pandas tqdm openpyxl pytesseract pillow PyMuPDF transformers torch sentence-transformers scikit-learn
```

## 🚀 Χρήση

### **Βασική Χρήση**
```bash
# Ανάλυση τρέχοντος φακέλου
python3 super_file_organizer.py

# Ανάλυση συγκεκριμένου φακέλου
python3 super_file_organizer.py --path "/path/to/your/files"

# Custom output file
python3 super_file_organizer.py --path "/path/to/files" --output "my_analysis.xlsx"
```

### **Προηγμένη Χρήση**
```bash
# Με verbose logging
python3 super_file_organizer.py --path "/path/to/files" --verbose

# Custom configuration
python3 super_file_organizer.py --config my_config.json

# Μόνο installation dependencies
python3 super_file_organizer.py --install-deps

# Help
python3 super_file_organizer.py --help
```

## ⚙️ Configuration

Το script δημιουργεί αυτόματα `super_file_organizer_config.json`:

```json
{
  "game_names": ["queena", "dolphin", "adventure", "puzzle", "casino"],
  "containers": ["Container 1", "Container 2", "Container 3"],
  "suppliers": ["Supplier A", "Supplier B", "Supplier C"],
  "years": ["2024", "2025"],
  "excluded_extensions": [".DS_Store", ".thumbs.db", ".tmp", ".log"],
  "max_file_size_mb": 500,
  "confidence_threshold": 0.7,
  "enable_ai_classification": true,
  "enable_ocr": true,
  "enable_semantic_matching": true,
  "parallel_processing": true
}
```

### **Προσαρμογή για το Business σας**
```json
{
  "game_names": ["Slot Gold", "Blackjack Pro", "Roulette Master"],
  "containers": ["Gaming Machines", "Table Games", "Software"],
  "suppliers": ["IGT", "Novomatic", "Aristocrat"],
  "years": ["2023", "2024", "2025"]
}
```

## 📊 Output

Το script δημιουργεί **Excel file** με multiple sheets:

### **1. All_Files** - Όλα τα αρχεία
- Current Path, File Name, Document Type
- Classification Confidence, AI Enhanced
- Suggested Game, Recommended Action
- **Suggested_New_Path** - Προτεινόμενη οργάνωση

### **2. Requires_Review** - Αρχεία για έλεγχο
- Low confidence classifications
- Temporary files
- Unclassified files

### **3. Duplicates** - Διπλότυπα αρχεία
- Grouped by hash
- Duplicate detection με confidence

### **4. AI_Enhanced** - AI Analysis
- OCR extracted text samples
- AI classification results
- Content analysis confidence

### **5. Statistics** - Στατιστικά
- Processing metrics
- AI performance stats
- Safety verification results

## 🎯 Supported File Types

### **Documents**
- **CE Certificates** - PDF με πιστοποιητικά συμμόρφωσης
- **Manuals** - PDF, DOC, DOCX εγχειρίδια
- **Invoices** - PDF, JPG τιμολόγια
- **Bank Proofs** - PDF, JPG αποδείξεις πληρωμής
- **Packing Lists** - PDF, JPG λίστες αποστολής

### **Languages**
- **English** - Full support
- **Greek (Ελληνικά)** - Πλήρης υποστήριξη
- **Mixed** - Αναγνώριση μικτών αρχείων

## 🛡️ Ασφάλεια

### **Zero-Loss Guarantee**
- ✅ **Δεν αλλάζει** αρχεία
- ✅ **Δεν μετακινεί** αρχεία
- ✅ **Δεν διαγράφει** αρχεία
- ✅ **Read-only** analysis μόνο
- ✅ **Safety verification** σε κάθε βήμα

### **Error Handling**
- Robust error recovery
- Detailed logging
- Graceful failures
- Progress tracking

## 🚨 Troubleshooting

### **OCR Issues**
```bash
# Έλεγχος tesseract installation
tesseract --version

# Έλεγχος διαθέσιμων γλωσσών
tesseract --list-langs
```

### **Memory Issues**
Στο config, μειώστε:
```json
{
  "max_file_size_mb": 100,
  "hash_cache_size": 5000,
  "parallel_processing": false
}
```

### **Performance Tuning**
```json
{
  "parallel_processing": true,
  "max_workers": 8,
  "chunk_size": 1000,
  "enable_fast_mode": true
}
```

## 📈 Performance

- **Small folders** (< 100 files): ~1-2 seconds
- **Medium folders** (100-1000 files): ~10-30 seconds
- **Large folders** (1000+ files): ~1-5 minutes
- **AI Enhancement**: +20-50% time για PDF analysis

## 🔄 Workflow

1. **Scan** - Ανάλυση αρχείων
2. **Classify** - AI + Pattern classification
3. **Detect Duplicates** - Hash-based detection
4. **Generate Suggestions** - Προτάσεις οργάνωσης
5. **Export Excel** - Αναλυτικό report
6. **Review** - Έλεγχος προτάσεων
7. **Execute** - Εφαρμογή (manual ή script)

## 🤝 Contributing

Το script είναι fully functional! Για βελτιώσεις:
- Custom AI models
- Additional file types
- More languages
- Custom export formats

## 📄 License

MIT License - Free to use and modify

---

**🎯 Έτοιμο για production use!**
**🔥 AI-powered, Zero-loss, Supercharged!**