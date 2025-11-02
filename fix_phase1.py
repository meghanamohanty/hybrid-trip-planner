#!/usr/bin/env python3
"""
Quick Fix for Phase 1 Import Issues
===================================

This script fixes the import problems discovered in testing:
1. Removes missing logger import from utils/__init__.py
2. Creates a simple logger.py stub if needed
3. Fixes any other import issues

Run this before running test_phase1.py again.
"""

import os
import sys


def fix_utils_init():
    """Fix the utils/__init__.py file to remove logger import"""
    utils_init_path = "src/utils/__init__.py"
    
    if not os.path.exists(utils_init_path):
        print(f"❌ Could not find {utils_init_path}")
        return False
    
    try:
        # Read current content
        with open(utils_init_path, 'r') as f:
            content = f.read()
        
        # Remove the problematic logger import line
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip the logger import line
            if 'from .logger import Logger' in line:
                print(f"✅ Removed problematic line: {line.strip()}")
                continue
            # Also remove any Logger usage in __all__
            if '"Logger"' in line or "'Logger'" in line:
                continue
            fixed_lines.append(line)
        
        # Write fixed content
        fixed_content = '\n'.join(fixed_lines)
        
        with open(utils_init_path, 'w') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed {utils_init_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {utils_init_path}: {e}")
        return False


def create_missing_files():
    """Create any missing utility files that are imported"""
    
    # Create a simple stub for logger.py if it's still needed
    logger_path = "src/utils/logger.py"
    if not os.path.exists(logger_path):
        logger_content = '''"""
Simple Logger Stub
=================
Basic logger implementation if needed.
"""

import logging

class Logger:
    """Simple logger wrapper"""
    
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
'''
        
        try:
            with open(logger_path, 'w') as f:
                f.write(logger_content)
            print(f"✅ Created {logger_path}")
        except Exception as e:
            print(f"❌ Error creating {logger_path}: {e}")


def check_project_structure():
    """Check if we're in the right directory"""
    required_dirs = ["src", "src/utils", "src/data_pipeline"]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("Make sure you're running this from the project root directory")
        return False
    
    print("✅ Project structure looks correct")
    return True


def fix_test_script():
    """Fix the test script to handle import errors better"""
    test_path = "test_phase1.py"
    
    if not os.path.exists(test_path):
        print(f"❌ Could not find {test_path}")
        return False
    
    try:
        with open(test_path, 'r') as f:
            content = f.read()
        
        # Fix the package checking logic
        fixed_content = content.replace(
            '__import__(package.replace("-", "_"))',
            '__import__(package.replace("-", "_").replace("python_dotenv", "dotenv"))'
        )
        
        with open(test_path, 'w') as f:
            f.write(fixed_content)
        
        print("✅ Fixed test script package detection")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing test script: {e}")
        return False


def main():
    """Run all fixes"""
    print("🔧 Running Quick Fix for Phase 1 Import Issues")
    print("=" * 50)
    
    # Check if we're in the right place
    if not check_project_structure():
        print("\n❌ Please run this script from your project root directory")
        print("   (where src/ directory is located)")
        return False
    
    # Apply fixes
    fixes_applied = 0
    
    if fix_utils_init():
        fixes_applied += 1
    
    create_missing_files()
    fixes_applied += 1
    
    if fix_test_script():
        fixes_applied += 1
    
    print(f"\n✅ Applied {fixes_applied} fixes")
    print("\nNow try running the test again:")
    print("python test_phase1.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)