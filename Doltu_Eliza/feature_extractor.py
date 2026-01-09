"""
Feature extraction for spam detection.
Extracts various spam indicators from email content.
"""
import re
from utils import EmailParser


class FeatureExtractor:
    """Extract spam-indicative features from email content."""
    
    # Common spam keywords
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
        'rolex', 'replica', 'watches', 'luxury', 'designer'
    }
    
    # Suspicious phrases
    SPAM_PHRASES = [
        'click here', 'click below', 'click now', 'click this',
        'act now', 'limited time', 'hurry up', 'don\'t wait',
        'free money', 'make money', 'extra income', 'work from home',
        'risk free', 'no risk', 'money back', 'guarantee',
        'dear friend', 'dear sir', 'dear customer',
        'you have won', 'you are a winner', 'you have been selected',
        'claim your', 'claim now', 'collect your prize',
        'verify your account', 'confirm your', 'update your account',
        'suspended account', 'unusual activity', 'verify identity'
    ]
    
    @staticmethod
    def extract_features(content):
        """
        Extract all features from email content.
        Returns a dictionary of feature scores.
        """
        subject, body = EmailParser.parse_email(content)
        full_text = content.lower()
        
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
        }
        
        return features
    
    @staticmethod
    def count_urls(text):
        """Count number of URLs in text."""
        # Match http://, https://, www., and common domain patterns
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[a-zA-Z0-9-]+\.(com|net|org|info|biz|co|uk|ru|cn)[^\s]*'
        ]
        
        count = 0
        for pattern in url_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return min(count, 20)  # Cap at 20 to avoid outliers
    
    @staticmethod
    def check_excessive_caps(text):
        """Check for excessive capitalization (spam indicator)."""
        if not text:
            return 0
        
        # Remove whitespace for calculation
        text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
        
        if len(text_no_space) < 10:
            return 0
        
        caps_count = sum(1 for c in text_no_space if c.isupper())
        caps_ratio = caps_count / len(text_no_space)
        
        # Score based on ratio
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
        
        # Count exclamation marks and question marks
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
        
        # Look for repeated punctuation
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
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count += len(re.findall(pattern, text))
        
        return min(count, 50)  # Cap at 50
    
    @staticmethod
    def count_spam_phrases(text):
        """Count spam phrases in text."""
        text = text.lower()
        count = 0
        
        for phrase in FeatureExtractor.SPAM_PHRASES:
            count += text.count(phrase)
        
        return min(count, 20)  # Cap at 20
    
    @staticmethod
    def check_suspicious_html(content):
        """Check for suspicious HTML patterns."""
        content_lower = content.lower()
        score = 0
        
        # Check for script tags
        if '<script' in content_lower:
            score += 10
        
        # Check for iframe
        if '<iframe' in content_lower:
            score += 8
        
        # Check for hidden text attempts
        if 'display:none' in content_lower or 'display: none' in content_lower:
            score += 5
        if 'visibility:hidden' in content_lower or 'visibility: hidden' in content_lower:
            score += 5
        
        # Check for font size 0 or very small
        if re.search(r'font-size:\s*0', content_lower):
            score += 5
        if re.search(r'font-size:\s*1px', content_lower):
            score += 3
        
        # Check for suspicious color matches (text same color as background)
        if 'color:#ffffff' in content_lower.replace(' ', '') and 'background:#ffffff' in content_lower.replace(' ', ''):
            score += 5
        
        # Check for excessive HTML tags
        html_tag_count = len(re.findall(r'<[^>]+>', content))
        if html_tag_count > 100:
            score += 5
        
        return min(score, 30)
    
    @staticmethod
    def score_subject(subject):
        """Score the subject line for spam indicators."""
        if not subject:
            return 0
        
        subject_lower = subject.lower()
        score = 0
        
        # Check for spam keywords in subject
        for keyword in ['free', 'winner', 'urgent', 'act now', 'limited', 'offer']:
            if keyword in subject_lower:
                score += 5
        
        # Check for excessive caps in subject
        if subject:
            caps_ratio = sum(1 for c in subject if c.isupper()) / len(subject)
            if caps_ratio > 0.5:
                score += 10
            elif caps_ratio > 0.3:
                score += 5
        
        # Check for excessive punctuation in subject
        if subject.count('!') > 1:
            score += 3
        if '!!!' in subject:
            score += 5
        
        # Check for "Re:" or "Fwd:" - less likely to be spam
        if subject_lower.startswith('re:') or subject_lower.startswith('fwd:'):
            score -= 5
        
        return max(score, 0)
    
    @staticmethod
    def count_numbers(text):
        """Count numbers in text (spam often has many numbers)."""
        numbers = re.findall(r'\b\d+\b', text)
        return min(len(numbers), 30)
    
    @staticmethod
    def count_currency_symbols(text):
        """Count currency symbols like $, €, £."""
        count = text.count('$') + text.count('€') + text.count('£') + text.count('¥')
        return min(count, 15)
    
    @staticmethod
    def count_email_addresses(text):
        """Count email addresses in text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return min(len(emails), 10)
    
    @staticmethod
    def compute_total_score(features):
        """
        Compute a total spam score from all features.
        Returns a score between 0 and 100.
        """
        # Weighted scoring of features
        score = 0
        
        score += features['url_count'] * 2.5
        score += features['excessive_caps'] * 1.5
        score += features['excessive_punctuation'] * 1.0
        score += features['spam_keywords_count'] * 2.0
        score += features['spam_phrases_count'] * 3.0
        score += features['suspicious_html'] * 1.5
        score += features['subject_spam_score'] * 2.0
        score += features['number_count'] * 0.5
        score += features['currency_symbols'] * 2.0
        score += features['email_addresses'] * 1.5
        
        # Normalize to 0-100 range
        # Empirically, spam scores tend to be > 50, clean < 30
        return min(score, 100)

