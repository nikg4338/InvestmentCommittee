#!/usr/bin/env python3
"""
Simple Emoji Fix
===============

Replace emojis in specific logging lines without touching indentation.
"""

import re

def fix_file_emojis(filepath):
    """Fix emojis in a specific file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Simple regex replacements for specific emoji patterns in logging
        patterns = [
            (r'logger\.info\("🚀([^"]*)"', r'logger.info("Starting\1"'),
            (r'logger\.info\("🔍([^"]*)"', r'logger.info("Scanning\1"'),
            (r'logger\.info\("🎯([^"]*)"', r'logger.info("EXECUTING\1"'),
            (r'logger\.info\("✅([^"]*)"', r'logger.info("SUCCESS:\1"'),
            (r'logger\.info\("📊([^"]*)"', r'logger.info("STATUS:\1"'),
            (r'logger\.error\("❌([^"]*)"', r'logger.error("ERROR:\1"'),
            (r'logger\.warning\("⚠️([^"]*)"', r'logger.warning("WARNING:\1"'),
            (r'logger\.info\("🏁([^"]*)"', r'logger.info("COMPLETE:\1"'),
            (r'logger\.info\("🔌([^"]*)"', r'logger.info("Connecting\1"'),
            
            # Handle f-strings
            (r'logger\.info\(f"🚀([^"]*)"', r'logger.info(f"Starting\1"'),
            (r'logger\.info\(f"🔍([^"]*)"', r'logger.info(f"Scanning\1"'),
            (r'logger\.info\(f"🎯([^"]*)"', r'logger.info(f"EXECUTING\1"'),
            (r'logger\.info\(f"✅([^"]*)"', r'logger.info(f"SUCCESS:\1"'),
            (r'logger\.info\(f"📊([^"]*)"', r'logger.info(f"STATUS:\1"'),
            (r'logger\.error\(f"❌([^"]*)"', r'logger.error(f"ERROR:\1"'),
            
            # Handle print statements
            (r'print\("\\n⚠️([^"]*)"', r'print("\\nWARNING:\1"'),
            (r'print\(f"\\n❌([^"]*)"', r'print(f"\\nERROR:\1"'),
            (r'print\("\\n🏁([^"]*)"', r'print("\\nCOMPLETE:\1"'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed emojis in {filepath}")
            return True
        else:
            print(f"No emojis to fix in {filepath}")
            return False
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix emoji issues in key files."""
    files = [
        'autonomous_trading_launcher.py',
        'trading/autonomous_committee.py', 
        'trading/trade_closer.py',
        'trading/real_alpaca_executor.py'
    ]
    
    print("Applying careful emoji fixes...")
    for file in files:
        fix_file_emojis(file)
    print("Done!")

if __name__ == "__main__":
    main()
