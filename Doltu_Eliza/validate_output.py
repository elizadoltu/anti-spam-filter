#!/usr/bin/env python3
"""
Validate that output format is correct according to requirements.
"""
import sys
import re


def validate_output_format(output_file):
    """
    Validate output format according to requirements:
    - filename|cln or filename|inf
    - No spaces around pipe
    - No full paths
    - Only cln or inf verdicts
    """
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    errors = []
    valid_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Check basic format
        if '|' not in line:
            errors.append(f"Line {i}: Missing pipe separator: {line}")
            continue
        
        parts = line.split('|')
        if len(parts) != 2:
            errors.append(f"Line {i}: Should have exactly one pipe: {line}")
            continue
        
        filename, verdict = parts
        
        # Check for spaces
        if ' ' in filename:
            errors.append(f"Line {i}: Filename contains space: '{filename}'")
        
        if filename != filename.strip():
            errors.append(f"Line {i}: Filename has leading/trailing spaces: '{filename}'")
        
        if verdict != verdict.strip():
            errors.append(f"Line {i}: Verdict has leading/trailing spaces: '{verdict}'")
        
        # Check for full path
        if '/' in filename or '\\' in filename:
            errors.append(f"Line {i}: Filename contains path separator (should be filename only): '{filename}'")
        
        # Check verdict
        if verdict not in ['cln', 'inf']:
            errors.append(f"Line {i}: Invalid verdict '{verdict}' (must be 'cln' or 'inf')")
        
        if not errors or len(errors) == valid_count:
            valid_count += 1
    
    return errors, valid_count, len(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_output.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    print(f"Validating output format: {output_file}")
    print("="*60)
    
    errors, valid_count, total_lines = validate_output_format(output_file)
    
    if errors:
        print(f"\n❌ VALIDATION FAILED - {len(errors)} error(s) found:\n")
        for error in errors:
            print(f"  {error}")
    else:
        print(f"\n✅ VALIDATION PASSED")
    
    print(f"\nSummary:")
    print(f"  Total lines: {total_lines}")
    print(f"  Valid lines: {valid_count}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

