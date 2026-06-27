#!/usr/bin/env python3
"""rename_to_eml.py - Rename extensionless email files to .eml"""

import os
from pathlib import Path

def is_email_file(file_path):
    """Check if a file looks like an email."""
    try:
        with open(file_path, "rb") as f:
            first_line = f.readline().decode("utf-8", errors="ignore").strip().lower()
            return any(first_line.startswith(prefix) for prefix in [
                "from ", "received:", "return-path:", "date:", "message-id:",
                "delivered-to:", "x-", "subject:", "to:", "mime-version:",
                "content-type:", "content-transfer-encoding:"
            ])
    except Exception:
        return False

def rename_folder(input_folder, recursive=False):
    folder = Path(input_folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    
    if recursive:
        files = sorted([f for f in folder.rglob("*") if f.is_file()])
    else:
        files = sorted([f for f in folder.glob("*") if f.is_file()])
    
    # Filter: only files without .eml extension
    non_eml = [f for f in files if f.suffix.lower() != ".eml"]
    
    if not non_eml:
        print("All files already have .eml extension or no files found.")
        return
    
    print(f"Found {len(non_eml)} files without .eml extension.")
    
    renamed = 0
    skipped = 0
    
    for file_path in non_eml:
        if is_email_file(file_path):
            new_path = file_path.with_suffix(file_path.suffix + ".eml")
            # If the file already has a suffix, keep it (e.g., file.txt -> file.txt.eml)
            try:
                file_path.rename(new_path)
                renamed += 1
                if renamed % 100 == 0:
                    print(f"  Renamed {renamed} files...")
            except Exception as e:
                print(f"  Error renaming {file_path.name}: {e}")
                skipped += 1
        else:
            skipped += 1
    
    print(f"\nDone. Renamed: {renamed}, Skipped: {skipped}")
    print(f"Folder: {folder.resolve()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rename email files to .eml")
    parser.add_argument("input_folder", help="Folder containing email files")
    parser.add_argument("--recursive", action="store_true", help="Process subfolders")
    args = parser.parse_args()
    rename_folder(args.input_folder, args.recursive)
