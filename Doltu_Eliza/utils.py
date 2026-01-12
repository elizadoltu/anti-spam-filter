"""
Utility functions for email parsing and file handling.
"""
import chardet
import re
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
        
        try:
            encoding = EmailParser.detect_encoding(file_path)
        except:
            encoding = 'utf-8'
        
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    content = f.read()
                return content
            except:
                continue
        
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
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def tokenize(text):
        """
        Enhanced tokenization with better text preprocessing.
        Removes HTML, normalizes text, filters stopwords.
        """
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', ' ', text)
        text = re.sub(r'www\.[^\s]+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        
        # Replace common separators with spaces
        for char in '.,;:!?()[]{}"\'\\n\\r\\t':
            text = text.replace(char, ' ')
        
        # Extract words (2-15 characters, alphabetic only)
        words = re.findall(r'\b[a-z]{2,15}\b', text)
        
        # Basic stopwords (you can expand this list)
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that',
            'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can'
        }
        
        # Filter out stopwords and very common words
        words = [w for w in words if w not in stopwords]
        
        return words