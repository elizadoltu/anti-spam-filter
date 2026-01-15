"""
Feature extraction for spam detection.
Extracts various spam indicators from email content.
"""
import re
from utils import EmailParser
from ngram_detector import NGramDetector

class FeatureExtractor:
    """Extract spam-indicative features from email content."""
    
    # Expanded spam keywords
    SPAM_KEYWORDS = {
        'free', 'winner', 'cash', 'prize', 'money', 'credit', 'earn', 'income',
        'million', 'dollars', 'lottery', 'congratulations', 'click', 'here',
        'urgent', 'limited', 'offer', 'buy', 'order', 'discount', 'cheap',
        'viagra', 'pharmacy', 'pills', 'weight', 'loss', 'debt', 'loan',
        'mortgage', 'refinance', 'guarantee', 'risk-free', 'bonus', 'gift',
        'subscription', 'unsubscribe', 'remove', 'opt-out', 'casino', 'gambling',
        'porn', 'xxx', 'sex', 'dating', 'singles', 'romance', 'meet',
        'investment', 'stock', 'profit', 'revenue', 'business', 'opportunity',
        'work from home', 'make money', 'extra income', 'financial freedom',
        'act now', 'don\'t wait', 'limited time', 'expires', 'hurry',
        'congratulation', 'selected', 'chosen', 'approved', 'qualified',
        'claim', 'redeem', 'collect', 'confirm', 'verify', 'account',
        'password', 'security', 'suspended', 'locked', 'unauthorized',
        'diploma', 'degree', 'university', 'pharmacy', 'medication',
        'prescription', 'cialis', 'levitra', 'xanax', 'valium',
        'rolex', 'replica', 'watches', 'luxury', 'designer',
        # Additional spam keywords
        'nigeria', 'prince', 'inheritance', 'beneficiary', 'transfer',
        'confidential', 'urgent reply', 'million dollars', 'bank account',
        'wire transfer', 'western union', 'moneygram', 'paypal', 'bitcoin',
        'cryptocurrency', 'forex', 'trading', 'binary', 'options',
        'mlm', 'multi-level', 'pyramid', 'scheme', 'get rich',
        'no experience', 'required', 'easy money', 'fast cash',
        'lowest price', 'best price', 'special promotion', 'trial',
        'sample', 'bargain', 'clearance', 'credit card', 'order now',
        'apply now', 'call now', 'subscribe', 'membership', 'teen',
        'adult', 'porn', 'webcam', 'live', 'chat', 'hot', 'sexy'
    }
    
    # Expanded suspicious phrases
    SPAM_PHRASES = [
        'click here', 'click below', 'click now', 'click this',
        'act now', 'limited time', 'hurry up', 'don\'t wait',
        'free money', 'make money', 'extra income', 'work from home',
        'risk free', 'no risk', 'money back', 'guarantee',
        'dear friend', 'dear sir', 'dear customer',
        'you have won', 'you are a winner', 'you have been selected',
        'claim your', 'claim now', 'collect your prize',
        'verify your account', 'confirm your', 'update your account',
        'suspended account', 'unusual activity', 'verify identity',
        'nigerian prince', 'million dollars', 'bank transfer',
        'wire transfer', 'urgent response', 'reply immediately',
        'act immediately', 'time sensitive', 'final notice',
        'order status', 'refund status', 'payment required',
        'credit card', 'social security', 'account number'
    ]
    
    @staticmethod
    def extract_features(content):
        """
        Extract all features from email content.
        Returns a dictionary of feature scores.
        """
        subject, body = EmailParser.parse_email(content)
        full_text = content.lower()
        
        # Tokenize for n-gram detection
        words = EmailParser.tokenize(full_text)
        
        features = {
            'url_count': FeatureExtractor.count_urls(full_text),
            'excessive_caps': FeatureExtractor.check_excessive_caps(content),
            'excessive_punctuation': FeatureExtractor.check_excessive_punctuation(content),
            'spam_keywords_count': FeatureExtractor.count_spam_keywords(full_text),
            'spam_phrases_count': FeatureExtractor.count_spam_phrases(full_text),
            'suspicious_html': FeatureExtractor.check_suspicious_html(content),
            'subject_spam_score': FeatureExtractor.score_subject(subject),
            'number_count': FeatureExtractor.count_numbers(full_text),
            'currency_symbols': FeatureExtractor.count_currency_symbols(content),
            'email_addresses': FeatureExtractor.count_email_addresses(full_text),
            'ngram_spam_score': NGramDetector.count_spam_phrases(words),  # NEW: N-gram phrases!
            # New advanced features
            'text_length': FeatureExtractor.analyze_length(content),
            'special_char_density': FeatureExtractor.special_char_density(content),
            'html_density': FeatureExtractor.html_density(content),
            'repeated_words': FeatureExtractor.count_repeated_words(content)
        }
        
        return features
    
    @staticmethod
    def count_urls(text):
        """Count number of URLs in text."""
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[a-zA-Z0-9-]+\.(com|net|org|info|biz|co|uk|ru|cn)[^\s]*'
        ]
        
        count = 0
        for pattern in url_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return min(count, 20)
    
    @staticmethod
    def check_excessive_caps(text):
        """Check for excessive capitalization (spam indicator)."""
        if not text:
            return 0
        
        text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
        if len(text_no_space) < 10:
            return 0
        
        caps_count = sum(1 for c in text_no_space if c.isupper())
        caps_ratio = caps_count / len(text_no_space)
        
        if caps_ratio > 0.5:
            return 10
        elif caps_ratio > 0.3:
            return 5
        elif caps_ratio > 0.2:
            return 2
        return 0
    
    @staticmethod
    def check_excessive_punctuation(text):
        """Check for excessive punctuation marks."""
        if not text:
            return 0
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        dollar_count = text.count('$')
        
        score = 0
        if exclamation_count > 5:
            score += min(exclamation_count, 10)
        if question_count > 3:
            score += min(question_count, 5)
        if dollar_count > 2:
            score += min(dollar_count * 2, 10)
        
        if re.search(r'!{2,}', text):
            score += 5
        if re.search(r'\?{2,}', text):
            score += 5
        if re.search(r'\${2,}', text):
            score += 5
        
        return min(score, 20)
    
    @staticmethod
    def count_spam_keywords(text):
        """Count spam keywords in text."""
        text = text.lower()
        count = 0
        for keyword in FeatureExtractor.SPAM_KEYWORDS:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count += len(re.findall(pattern, text))
        return min(count, 50)
    
    @staticmethod
    def count_spam_phrases(text):
        """Count spam phrases in text."""
        text = text.lower()
        count = 0
        for phrase in FeatureExtractor.SPAM_PHRASES:
            count += text.count(phrase)
        return min(count, 20)
    
    @staticmethod
    def check_suspicious_html(content):
        """Check for suspicious HTML patterns."""
        content_lower = content.lower()
        score = 0
        
        if '<script' in content_lower:
            score += 10
        
        if 'javascript:' in content_lower:
            score += 8
        
        if '<iframe' in content_lower:
            score += 8
        
        if 'onclick=' in content_lower or 'onload=' in content_lower:
            score += 5
        
        form_count = content_lower.count('<form')
        if form_count > 0:
            score += min(form_count * 5, 15)
        
        return min(score, 25)
    
    @staticmethod
    def score_subject(subject):
        """Score the subject line for spam indicators."""
        if not subject:
            return 0
        
        subject_lower = subject.lower()
        score = 0
        
        spam_subject_words = ['free', 'urgent', 'winner', 'congratulations', 
                             'alert', 'reminder', 're:', 'fwd:']
        for word in spam_subject_words:
            if word in subject_lower:
                score += 3
        
        if subject.isupper() and len(subject) > 10:
            score += 5
        
        exclamation_count = subject.count('!')
        if exclamation_count > 0:
            score += min(exclamation_count * 2, 8)
        
        return min(score, 15)
    
    @staticmethod
    def count_numbers(text):
        """Count numeric sequences in text."""
        numbers = re.findall(r'\b\d+\b', text)
        return min(len(numbers), 30)
    
    @staticmethod
    def count_currency_symbols(text):
        """Count currency symbols."""
        currency_symbols = ['$', '€', '£', '¥']
        count = sum(text.count(symbol) for symbol in currency_symbols)
        return min(count, 15)
    
    @staticmethod
    def count_email_addresses(text):
        """Count email addresses in text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return min(len(emails), 10)
    
    @staticmethod
    def analyze_length(content):
        """Analyze text length patterns."""
        length = len(content)
        word_count = len(content.split())
        
        # Very short emails with URLs are often spam
        if length < 200 and 'http' in content.lower():
            return 5
        
        # Very long emails with high keyword density
        if length > 5000:
            return 3
        
        # Normal length
        return 0
    
    @staticmethod
    def special_char_density(content):
        """Calculate special character density."""
        if not content:
            return 0
        
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        density = special_chars / len(content)
        
        if density > 0.15:
            return 8
        elif density > 0.10:
            return 4
        return 0
    
    @staticmethod
    def html_density(content):
        """Calculate HTML tag density."""
        html_tags = len(re.findall(r'<[^>]+>', content))
        if len(content) == 0:
            return 0
        
        density = html_tags / max(len(content), 1) * 1000
        
        if density > 20:
            return 10
        elif density > 10:
            return 5
        return 0
    
    @staticmethod
    def count_repeated_words(content):
        """Count repeated words (spam often repeats keywords)."""
        words = content.lower().split()
        if not words:
            return 0
        
        word_counts = {}
        for word in words:
            if len(word) > 4:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated = sum(1 for count in word_counts.values() if count > 3)
        return min(repeated * 2, 10)
    
    @staticmethod
    def compute_total_score(features):
        """Compute total spam score from all features."""
        weights = {
            'url_count': 2.5,
            'excessive_caps': 1.5,
            'excessive_punctuation': 1.2,
            'spam_keywords_count': 2.0,
            'spam_phrases_count': 3.0,
            'ngram_spam_score': 3.5,  # NEW: N-grams are very important!
            'suspicious_html': 2.0,
            'subject_spam_score': 2.5,
            'number_count': 0.5,
            'currency_symbols': 1.5,
            'email_addresses': 1.0,
            'text_length': 1.0,
            'special_char_density': 1.5,
            'html_density': 1.8,
            'repeated_words': 1.3
        }
        
        total_score = 0
        for feature, value in features.items():
            weight = weights.get(feature, 1.0)
            total_score += value * weight
        
        return total_score
