#!/bin/bash

# ΤΟΥΜΠΑΝΗ File Organizer Run Script
# Quick launcher with proper directory setup

echo "🚀 Starting ΤΟΥΜΠΑΝΗ File Organizer..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "📁 Working directory: $PWD"

# Check if the Python file exists
if [ ! -f "super_file_organizer.py" ]; then
    echo "❌ Error: super_file_organizer.py not found in $PWD"
    exit 1
fi

# Run the file organizer
echo "▶️ Launching GUI..."
python3 super_file_organizer.py --gui

echo "✅ Finished!"