#!/usr/bin/env python3
"""
Quick Fix: Remove Emojis for Windows Compatibility
=====================================================

This script removes emojis from logging messages to prevent Unicode encoding errors
on Windows PowerShell.
"""

import re
import os

def remove_emojis_from_file(filepath):
    """Remove emojis from a Python file."""
    
    # Emoji replacements for logging messages
    replacements = {
        '🚀': '',
        '🔍': '',
        '🎯': 'EXECUTING',
        '✅': '',
        '📊': '',
        '❌': 'ERROR:',
        '⚠️': 'WARNING:',
        '🏁': '',
        '🔌': '',
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace emojis in logging messages
    for emoji, replacement in replacements.items():
        if emoji in content:
            # For logging messages, clean up the format
            if replacement:
                content = content.replace(f'"{emoji} ', f'"{replacement} ')
                content = content.replace(f"'{emoji} ", f"'{replacement} ")
                content = content.replace(f'f"{emoji} ', f'f"{replacement} ')
                content = content.replace(f"f'{emoji} ", f"f'{replacement} ")
            else:
                # Just remove the emoji and space
                content = content.replace(f'{emoji} ', '')
                content = content.replace(f'{emoji}', '')
    
    # Clean up double spaces
    content = re.sub(r'  +', ' ', content)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed emojis in {filepath}")
        return True
    else:
        print(f"- No emojis found in {filepath}")
        return False

def main():
    """Remove emojis from key trading files."""
    
    files_to_fix = [
        'autonomous_trading_launcher.py',
        'trading/autonomous_committee.py',
        'trading/trade_closer.py',
        'trading/real_alpaca_executor.py'
    ]
    
    print("Fixing emoji encoding issues for Windows compatibility...")
    print("=" * 60)
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            remove_emojis_from_file(filepath)
        else:
            print(f"- File not found: {filepath}")
    
    print("=" * 60)
    print("Emoji fix complete! The system should now run without Unicode errors.")

if __name__ == "__main__":
    main()
