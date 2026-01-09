"""
Utility functions for email parsing and file handling.
"""
import chardet
from pathlib import Path


class EmailParser:
    """Handles reading and parsing email files with various encodings."""
    
    @staticmethod
    def detect_encoding(file_path):
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] if result['encoding'] else 'utf-8'
    
    @staticmethod
    def read_email(file_path):
        """
        Read an email file and return its content.
        First line is the subject, rest is the body.
        """
        file_path = Path(file_path)
        
        # Try to detect encoding
        try:
            encoding = EmailParser.detect_encoding(file_path)
        except:
            encoding = 'utf-8'
        
        # Try multiple encodings if the detected one fails
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    content = f.read()
                return content
            except:
                continue
        
        # Last resort: read as binary and decode with errors ignored
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        return content
    
    @staticmethod
    def parse_email(content):
        """
        Parse email content into subject and body.
        First line is subject, rest is body.
        """
        lines = content.split('\n', 1)
        subject = lines[0].strip() if lines else ""
        body = lines[1] if len(lines) > 1 else ""
        
        return subject, body
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text for processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def tokenize(text):
        """
        Tokenize text into words.
        Simple word splitting with basic cleanup.
        """
        # Remove common punctuation but keep some meaningful chars
        text = text.lower()
        
        # Replace common separators with spaces
        for char in '.,;:!?()[]{}"\'\n\r\t':
            text = text.replace(char, ' ')
        
        # Split into words
        words = text.split()
        
        # Filter out very short words and numbers
        words = [w for w in words if len(w) > 2 and not w.isdigit()]
        
        return words

