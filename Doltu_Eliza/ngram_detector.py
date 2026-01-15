"""
N-gram (phrase) detector for spam.
"""
from collections import Counter

class NGramDetector:
    """Detects spam using bigrams (2-word phrases) and trigrams (3-word phrases)."""
    
    # Common spam bigrams (2-word phrases)
    SPAM_BIGRAMS = {
        ('click', 'here'), ('click', 'now'), ('act', 'now'), ('buy', 'now'),
        ('free', 'money'), ('make', 'money'), ('earn', 'money'), ('get', 'money'),
        ('take', 'money'), ('win', 'money'), ('extra', 'income'), ('easy', 'money'),
        ('limited', 'time'), ('limited', 'offer'), ('hurry', 'up'), ('dont', 'wait'),
        ('free', 'gift'), ('free', 'trial'), ('risk', 'free'), ('money', 'back'),
        ('no', 'risk'), ('guaranteed', 'results'), ('instant', 'access'),
        ('work', 'from'), ('from', 'home'), ('work', 'home'),
        ('lose', 'weight'), ('weight', 'loss'), ('diet', 'pill'),
        ('bank', 'account'), ('credit', 'card'), ('social', 'security'),
        ('verify', 'account'), ('confirm', 'identity'), ('update', 'information'),
        ('dear', 'friend'), ('dear', 'customer'), ('congratulations', 'you'),
        ('you', 'won'), ('youve', 'won'), ('you', 'are'), ('selected', 'winner'),
        ('claim', 'prize'), ('claim', 'your'), ('collect', 'prize'),
        ('million', 'dollars'), ('cash', 'prize'), ('lottery', 'winner'),
    }
    
    # Common spam trigrams (3-word phrases)
    SPAM_TRIGRAMS = {
        ('click', 'here', 'now'), ('act', 'now', 'limited'),
        ('work', 'from', 'home'), ('make', 'money', 'fast'),
        ('lose', 'weight', 'fast'), ('earn', 'extra', 'income'),
        ('limited', 'time', 'offer'), ('free', 'money', 'now'),
        ('congratulations', 'you', 'won'), ('claim', 'your', 'prize'),
    }
    
    @staticmethod
    def extract_bigrams(words):
        """Extract all bigrams (2-word phrases) from a list of words."""
        return [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    @staticmethod
    def extract_trigrams(words):
        """Extract all trigrams (3-word phrases) from a list of words."""
        return [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    
    @staticmethod
    def count_spam_phrases(text_words):
        """
        Count how many spam bigrams and trigrams appear in the text.
        Returns a score based on phrase frequency.
        """
        if not text_words or len(text_words) < 2:
            return 0
        
        bigrams = NGramDetector.extract_bigrams(text_words)
        trigrams = NGramDetector.extract_trigrams(text_words) if len(text_words) >= 3 else []
        
        score = 0
        
        # Count spam bigrams (worth 3 points each)
        for bigram in bigrams:
            if bigram in NGramDetector.SPAM_BIGRAMS:
                score += 3
        
        # Count spam trigrams (worth 5 points each - more specific)
        for trigram in trigrams:
            if trigram in NGramDetector.SPAM_TRIGRAMS:
                score += 5
        
        return min(score, 50)  # Cap at 50
    
    @staticmethod
    def get_top_phrases(text_words, n=5):
        """
        Get the most common phrases in text.
        Useful for debugging/analysis.
        """
        if len(text_words) < 2:
            return []
        
        bigrams = NGramDetector.extract_bigrams(text_words)
        bigram_counts = Counter(bigrams)
        return bigram_counts.most_common(n)

