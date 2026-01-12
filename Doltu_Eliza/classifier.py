"""
Spam Classifier using Naive Bayes and feature-based detection.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from utils import EmailParser
from feature_extractor import FeatureExtractor

class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier for spam detection."""
    
    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.clean_word_counts = defaultdict(int)
        self.spam_total_words = 0
        self.clean_total_words = 0
        self.spam_doc_count = 0
        self.clean_doc_count = 0
        self.vocabulary = set()
        self.alpha = 1.0
    
    def train(self, clean_emails, spam_emails):
        """
        Train the classifier on clean and spam emails.
        Args:
            clean_emails: List of clean email texts
            spam_emails: List of spam email texts
        """
        for email in clean_emails:
            words = EmailParser.tokenize(email)
            self.clean_doc_count += 1
            for word in words:
                self.clean_word_counts[word] += 1
                self.clean_total_words += 1
                self.vocabulary.add(word)
        
        for email in spam_emails:
            words = EmailParser.tokenize(email)
            self.spam_doc_count += 1
            for word in words:
                self.spam_word_counts[word] += 1
                self.spam_total_words += 1
                self.vocabulary.add(word)
    
    def get_word_probability(self, word, is_spam):
        """Calculate P(word|spam) or P(word|clean) with Laplace smoothing."""
        if is_spam:
            word_count = self.spam_word_counts[word]
            total_words = self.spam_total_words
        else:
            word_count = self.clean_word_counts[word]
            total_words = self.clean_total_words
        
        vocab_size = len(self.vocabulary)
        probability = (word_count + self.alpha) / (total_words + self.alpha * vocab_size)
        return probability
    
    def calculate_log_probability(self, text, is_spam):
        """Calculate log probability of text being spam or clean."""
        words = EmailParser.tokenize(text)
        
        total_docs = self.spam_doc_count + self.clean_doc_count
        if total_docs == 0:
            prior = 0.5
        else:
            if is_spam:
                prior = self.spam_doc_count / total_docs
            else:
                prior = self.clean_doc_count / total_docs
        
        log_prob = math.log(prior) if prior > 0 else -100
        
        for word in words:
            word_prob = self.get_word_probability(word, is_spam)
            log_prob += math.log(word_prob)
        
        return log_prob
    
    def predict(self, text):
        """Predict if text is spam or clean."""
        if self.spam_doc_count == 0 or self.clean_doc_count == 0:
            return False, 0.5
        
        spam_log_prob = self.calculate_log_probability(text, is_spam=True)
        clean_log_prob = self.calculate_log_probability(text, is_spam=False)
        
        max_log_prob = max(spam_log_prob, clean_log_prob)
        spam_prob = math.exp(spam_log_prob - max_log_prob)
        clean_prob = math.exp(clean_log_prob - max_log_prob)
        
        total_prob = spam_prob + clean_prob
        spam_probability = spam_prob / total_prob if total_prob > 0 else 0.5
        
        return spam_probability > 0.5, spam_probability

class SpamClassifier:
    """Hybrid spam classifier combining Naive Bayes with feature-based detection."""
    
    def __init__(self):
        self.naive_bayes = NaiveBayesClassifier()
        self.feature_threshold = 45  # Lowered from 52
        self.nb_weight = 0.55  # Reduced from 0.72
        self.feature_weight = 0.45  # Increased from 0.28
        self.trained = False
    
    def train_from_folders(self, clean_folder, spam_folder):
        """Train the classifier from folders containing clean and spam emails."""
        clean_folder = Path(clean_folder)
        spam_folder = Path(spam_folder)
        
        clean_emails = []
        if clean_folder.exists():
            for file_path in clean_folder.iterdir():
                if file_path.is_file():
                    try:
                        email_content = EmailParser.read_email(file_path)
                        clean_emails.append(email_content)
                    except:
                        pass
        
        spam_emails = []
        if spam_folder.exists():
            for file_path in spam_folder.iterdir():
                if file_path.is_file():
                    try:
                        email_content = EmailParser.read_email(file_path)
                        spam_emails.append(email_content)
                    except:
                        pass
        
        if clean_emails and spam_emails:
            self.naive_bayes.train(clean_emails, spam_emails)
            self.trained = True
    
    def classify(self, email_content):
        """
        Classify an email as spam (True) or clean (False).
        Uses hybrid approach combining Naive Bayes and feature extraction.
        """
        features = FeatureExtractor.extract_features(email_content)
        feature_score = FeatureExtractor.compute_total_score(features)
        
        if not self.trained:
            return feature_score > self.feature_threshold
        
        is_spam_nb, spam_probability = self.naive_bayes.predict(email_content)
        
        normalized_feature_score = min(feature_score / 100.0, 1.0)
        
        combined_score = (self.nb_weight * spam_probability +
                         self.feature_weight * normalized_feature_score)
        
        # More aggressive threshold - lowered from 0.58 to 0.45
        threshold = 0.45
        
        # Adjusted heuristics for better detection
        if feature_score > 65 and spam_probability > 0.42:
            return True
        
        if feature_score < 25 and spam_probability < 0.50:
            return False
        
        # More aggressive spam detection on borderline cases
        if spam_probability > 0.48 and feature_score > 35:
            return True
        
        return combined_score > threshold
    
    def save_model(self, filepath):
        """Save the trained model to a JSON file."""
        model_data = {
            'spam_word_counts': dict(self.naive_bayes.spam_word_counts),
            'clean_word_counts': dict(self.naive_bayes.clean_word_counts),
            'spam_total_words': self.naive_bayes.spam_total_words,
            'clean_total_words': self.naive_bayes.clean_total_words,
            'spam_doc_count': self.naive_bayes.spam_doc_count,
            'clean_doc_count': self.naive_bayes.clean_doc_count,
            'vocabulary': list(self.naive_bayes.vocabulary),
            'feature_threshold': self.feature_threshold,
            'nb_weight': self.nb_weight,
            'feature_weight': self.feature_weight,
            'trained': self.trained
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.naive_bayes.spam_word_counts = defaultdict(int, model_data['spam_word_counts'])
        self.naive_bayes.clean_word_counts = defaultdict(int, model_data['clean_word_counts'])
        self.naive_bayes.spam_total_words = model_data['spam_total_words']
        self.naive_bayes.clean_total_words = model_data['clean_total_words']
        self.naive_bayes.spam_doc_count = model_data['spam_doc_count']
        self.naive_bayes.clean_doc_count = model_data['clean_doc_count']
        self.naive_bayes.vocabulary = set(model_data['vocabulary'])
        self.feature_threshold = model_data.get('feature_threshold', 45)
        self.nb_weight = model_data.get('nb_weight', 0.55)
        self.feature_weight = model_data.get('feature_weight', 0.45)
        self.trained = model_data.get('trained', False)
