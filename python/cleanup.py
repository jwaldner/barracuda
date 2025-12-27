#!/usr/bin/env python3
"""
Cleanup script for Barracuda project
Removes audit files, logs, and temporary data
"""

import os
import shutil
from pathlib import Path

def cleanup_barracuda():
    """Clean up audit files, logs, and temporary data"""
    
    # Get the project root (parent of python folder)
    project_root = Path(__file__).parent.parent
    
    # Files to delete
    files_to_delete = [
        project_root / "audit.json",
        project_root / "barracuda.log"
    ]
    
    # Folders to empty
    folders_to_empty = [
        project_root / "audits",
        project_root / "logs"
    ]
    
    print("🧹 Starting Barracuda cleanup...")
    
    # Delete individual files
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ Deleted: {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path.name}: {e}")
        else:
            print(f"⚪ Not found: {file_path.name}")
    
    # Empty folders
    for folder_path in folders_to_empty:
        if folder_path.exists() and folder_path.is_dir():
            try:
                # Remove all contents but keep the folder
                for item in folder_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        print(f"✅ Deleted: {folder_path.name}/{item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"✅ Deleted folder: {folder_path.name}/{item.name}")
                
                # Count remaining items
                remaining = list(folder_path.iterdir())
                if not remaining:
                    print(f"✅ Emptied: {folder_path.name}/ (0 items)")
                else:
                    print(f"⚠️  {folder_path.name}/ still has {len(remaining)} items")
                    
            except Exception as e:
                print(f"❌ Failed to empty {folder_path.name}/: {e}")
        else:
            print(f"⚪ Not found: {folder_path.name}/")
    
    print("🧹 Cleanup complete!")

if __name__ == "__main__":
    cleanup_barracuda()