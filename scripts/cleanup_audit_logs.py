#!/usr/bin/env python3
"""
Cleanup script to delete logs, audits, and audit.json files from the go-cuda project.
This script safely removes:
- All files in the logs/ directory
- All files in the audits/ directory  
- Any audit.json files in the project root
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_directory(dir_path):
    """Remove all files and subdirectories in the given directory."""
    if os.path.exists(dir_path):
        print(f"Cleaning up directory: {dir_path}")
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    print(f"  Deleted file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Deleted directory: {item}")
            except Exception as e:
                print(f"  Failed to delete {item}: {e}")
    else:
        print(f"Directory does not exist: {dir_path}")

def cleanup_audit_json_files(project_root):
    """Remove audit.json files from the project root."""
    audit_json_pattern = os.path.join(project_root, "audit.json*")
    audit_files = glob.glob(audit_json_pattern)
    
    if audit_files:
        print("Cleaning up audit.json files:")
        for file_path in audit_files:
            try:
                os.remove(file_path)
                print(f"  Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Failed to delete {os.path.basename(file_path)}: {e}")
    else:
        print("No audit.json files found to delete")

def cleanup_root_log_files(project_root):
    """Remove barracuda.log and other log files from the project root."""
    log_files = ["barracuda.log", "audit.json"]
    
    print("Cleaning up root log files:")
    for filename in log_files:
        file_path = os.path.join(project_root, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  Deleted: {filename}")
            except Exception as e:
                print(f"  Failed to delete {filename}: {e}")
        else:
            print(f"  File not found: {filename}")

def main():
    """Main cleanup function."""
    # Get the project root directory (assuming script is run from project root or scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if script_dir.endswith('scripts') else script_dir
    
    print(f"Project root: {project_root}")
    print("Starting cleanup process...\n")
    
    # Define directories to clean
    logs_dir = os.path.join(project_root, "logs")
    audits_dir = os.path.join(project_root, "audits")
    
    # Clean up directories
    cleanup_directory(logs_dir)
    cleanup_directory(audits_dir)
    
    # Clean up root log files (audit.json and barracuda.log)
    cleanup_root_log_files(project_root)
    
    print("\nCleanup completed!")

if __name__ == "__main__":
    main()