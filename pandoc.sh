#!/bin/bash
# Directory containing the EPUB files
SOURCE_DIR="./docs"
# Directory where the Markdown files will be saved
TARGET_DIR="./md"

# Function to process a single file
process_file() {
    epub="$1"
    filename=$(basename -- "$epub")
    base_name="${filename%.*}"
    # Create a directory for extracted media specific to this EPUB file
    mkdir -p "$TARGET_DIR/$base_name/img"
    # Convert EPUB to Markdown, placing extracted media in the created directory
    pandoc -t markdown --extract-media="$TARGET_DIR/$base_name/img" "$epub" --wrap=none -o "$TARGET_DIR/$base_name.md"
    echo "$filename converted."
}

export -f process_file
export SOURCE_DIR
export TARGET_DIR
# Find all EPUB files and pass them to parallel
# sudo apt-get install parallel
find "$SOURCE_DIR" -name "*.epub" | parallel process_file
echo "All conversions completed."