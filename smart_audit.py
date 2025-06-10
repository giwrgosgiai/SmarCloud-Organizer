import os
import datetime
import difflib
import hashlib
import pandas as pd
import subprocess
import json
from pathlib import Path

# Ρυθμίσεις
TARGET_DIR = "/Users/georgegiailoglou/Library/Mobile Documents/com~apple~CloudDocs/AdamsGames/1. Εγγραφα εταιριας"
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".png"}
MANDATORY_TYPES = ["CE", "Manual", "Invoice", "Packing List", "Bank Proof"]
TEMP_PATTERNS = ["temp", "copy", "draft", "check", "test"]
SKIPPED_FILES = []

# Βοηθητικό: απλό fingerprint αρχείου
def hash_file(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        SKIPPED_FILES.append({"file": filepath, "reason": str(e)})
        return None

# Ανιχνεύει αν το όνομα αρχείου δείχνει προσωρινό
def detect_temp_file(filename):
    name = filename.lower()
    return any(p in name for p in TEMP_PATTERNS)

# AI τύπος εγγράφου (με LLaMA)
def ai_guess_type(filepath, filename, container):
    prompt = f"""
Given the following information:
- File name: {filename}
- Folder name (container): {container}

What type of document is this? Possible types: Invoice, Manual, Packing List, CE, Bank Proof, or Other. Explain your guess in one sentence.
"""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3:instruct"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=20
        )
        output = result.stdout.decode("utf-8").strip()
        return output
    except Exception as e:
        return f"AI Error: {str(e)}"

# Κύρια συνάρτηση
results = []
now = datetime.datetime.now().strftime("%Y-%m-%d")
outfile = f"audit_report_{now}.xlsx"

for root, _, files in os.walk(TARGET_DIR):
    container = Path(root).name
    seen_hashes = {}

    for file in files:
        ext = Path(file).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        full_path = os.path.join(root, file)
        file_hash = hash_file(full_path)
        if file_hash is None:
            continue  # already logged

        is_temp = detect_temp_file(file)
        ai_analysis = ai_guess_type(full_path, file, container)

        # Ανίχνευση τύπου μέσω AI ή fallback
        if ":" in ai_analysis:
            ai_type, comment = ai_analysis.split(":", 1)
        else:
            ai_type, comment = "Unknown", ai_analysis

        # Ομαδοποίηση ίδιων αρχείων
        if file_hash in seen_hashes:
            status = "Duplicate"
            recommendation = "DELETE"
        else:
            status = "OK"
            recommendation = "KEEP"
            seen_hashes[file_hash] = file

        if is_temp:
            recommendation = "REVIEW"

        results.append({
            "Container": container,
            "File Name": file,
            "Document Type (AI)": ai_type.strip(),
            "AI Insight": comment.strip(),
            "Is_Temp": is_temp,
            "Recommendation": recommendation,
            "Path": full_path,
            "Hash": file_hash,
        })

# Εξαγωγή κύριων αποτελεσμάτων
df = pd.DataFrame(results)
df.to_excel(outfile, index=False)

# Καταγραφή skipped
if SKIPPED_FILES:
    skipped_df = pd.DataFrame(SKIPPED_FILES)
    with pd.ExcelWriter(outfile, mode="a", engine="openpyxl") as writer:
        skipped_df.to_excel(writer, index=False, sheet_name="Skipped Files")

print(f"✅ Audit ολοκληρώθηκε. Δημιουργήθηκε το αρχείο: {outfile}")