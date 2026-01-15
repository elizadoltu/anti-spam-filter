#!/usr/bin/env python3
"""
Anti-Spam Filter - Main CLI Entry Point
"""
import sys
import json
import os
from pathlib import Path

# Import our custom modules
from classifier import SpamClassifier
from utils import EmailParser


def write_info(output_file):
    """Write project information to output file in JSON format."""
    info = {
        "student_name": "Eliza Teodora Doltu",
        "project_name": "Anti-Spam Filter",
        "student_alias": "eliza.doltu",
        "project_version": "1.3.0"
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)


def scan_folder(folder_path, output_file, classifier):
    """
    Scan a folder for email files and classify them.
    Output format: filename|cln or filename|inf (one per line)
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    results = []
    
    # Get all files in the folder (non-recursive)
    files = [f for f in folder.iterdir() if f.is_file()]
    
    for file_path in sorted(files):
        try:
            # Parse the email
            email_text = EmailParser.read_email(file_path)
            
            # Classify the email
            is_spam = classifier.classify(email_text)
            
            # Format output: filename|verdict
            verdict = "inf" if is_spam else "cln"
            filename = file_path.name
            
            # Validate format (no spaces, just filename)
            if ' ' in filename:
                # This shouldn't happen in normal cases, but handle it
                filename = filename.replace(' ', '_')
            
            results.append(f"{filename}|{verdict}")
            
        except Exception as e:
            # If we can't process a file, mark it as clean by default
            print(f"Warning: Error processing {file_path.name}: {e}", file=sys.stderr)
            results.append(f"{file_path.name}|cln")
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 3:
        print("Usage:", file=sys.stderr)
        print("  main.py -info <output_file>", file=sys.stderr)
        print("  main.py -scan <folder> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "-info":
        if len(sys.argv) != 3:
            print("Error: -info requires exactly one argument: <output_file>", file=sys.stderr)
            sys.exit(1)
        output_file = sys.argv[2]
        write_info(output_file)
        
    elif command == "-scan":
        if len(sys.argv) != 4:
            print("Error: -scan requires exactly two arguments: <folder> <output_file>", file=sys.stderr)
            sys.exit(1)
        folder_path = sys.argv[2]
        output_file = sys.argv[3]
        
        # Initialize classifier
        classifier = SpamClassifier()
        
        # Try to load pre-trained model if it exists
        model_path = Path(__file__).parent / "trained_model.json"
        if model_path.exists():
            classifier.load_model(str(model_path))
        else:
            # If no trained model, use default rules-based approach
            print("Warning: No trained model found, using default classifier", file=sys.stderr)
        
        # Scan the folder
        scan_folder(folder_path, output_file, classifier)
        
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("Valid commands: -info, -scan", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

