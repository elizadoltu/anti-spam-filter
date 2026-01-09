#!/usr/bin/env python3
"""
Test script to evaluate classifier performance.
Usage: python test_classifier.py <model_file> <test_clean_folder> <test_spam_folder>
"""
import sys
from pathlib import Path
from classifier import SpamClassifier
from utils import EmailParser


def evaluate_folder(classifier, folder_path, expected_spam):
    """
    Evaluate classifier on a folder of emails.
    
    Args:
        classifier: SpamClassifier instance
        folder_path: Path to folder containing emails
        expected_spam: True if folder contains spam, False if clean
    
    Returns:
        tuple: (correct_count, total_count, predictions)
    """
    folder = Path(folder_path)
    correct = 0
    total = 0
    predictions = []
    
    if not folder.exists():
        return 0, 0, []
    
    for file_path in folder.iterdir():
        if file_path.is_file():
            try:
                email_content = EmailParser.read_email(file_path)
                is_spam = classifier.classify(email_content)
                
                total += 1
                if is_spam == expected_spam:
                    correct += 1
                
                predictions.append({
                    'file': file_path.name,
                    'predicted': 'spam' if is_spam else 'clean',
                    'expected': 'spam' if expected_spam else 'clean',
                    'correct': is_spam == expected_spam
                })
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
    
    return correct, total, predictions


def main():
    if len(sys.argv) < 4:
        print("Usage: python test_classifier.py <model_file> <test_clean_folder> <test_spam_folder>")
        sys.exit(1)
    
    model_file = sys.argv[1]
    clean_folder = sys.argv[2]
    spam_folder = sys.argv[3]
    
    print(f"Loading model from: {model_file}")
    
    # Load classifier
    classifier = SpamClassifier()
    try:
        classifier.load_model(model_file)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained classifier (feature-based only)")
    
    # Test on clean emails
    print(f"\nTesting on clean emails from: {clean_folder}")
    clean_correct, clean_total, clean_predictions = evaluate_folder(
        classifier, clean_folder, expected_spam=False
    )
    
    # Test on spam emails
    print(f"Testing on spam emails from: {spam_folder}")
    spam_correct, spam_total, spam_predictions = evaluate_folder(
        classifier, spam_folder, expected_spam=True
    )
    
    # Calculate metrics
    total_correct = clean_correct + spam_correct
    total_emails = clean_total + spam_total
    
    if total_emails > 0:
        accuracy = (total_correct / total_emails) * 100
    else:
        accuracy = 0
    
    # Detection rate (True Positive Rate)
    if spam_total > 0:
        detection_rate = (spam_correct / spam_total) * 100
    else:
        detection_rate = 0
    
    # False Positive Rate
    false_positives = clean_total - clean_correct
    if clean_total > 0:
        false_positive_rate = (false_positives / clean_total) * 100
    else:
        false_positive_rate = 0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nClean emails:")
    print(f"  Correct: {clean_correct}/{clean_total}")
    print(f"  False Positives: {false_positives} ({false_positive_rate:.2f}%)")
    
    print(f"\nSpam emails:")
    print(f"  Correct: {spam_correct}/{spam_total}")
    print(f"  Detection Rate: {detection_rate:.2f}%")
    
    print(f"\nOverall:")
    print(f"  Accuracy: {total_correct}/{total_emails} ({accuracy:.2f}%)")
    print(f"  Detection Rate: {detection_rate:.2f}%")
    print(f"  False Positive Rate: {false_positive_rate:.2f}%")
    
    # Calculate score like in the leaderboard
    # Score seems to be weighted: detection rate with penalty for false positives
    # Based on leaderboard, it looks like: Score ≈ Detection - FP_penalty
    score = detection_rate - (false_positive_rate * 5)  # Rough estimate
    print(f"  Estimated Score: {score:.2f}")
    
    print("="*60)
    
    # Show some misclassified examples
    print("\nMisclassified Clean Emails (False Positives):")
    fp_count = 0
    for pred in clean_predictions:
        if not pred['correct'] and fp_count < 5:
            print(f"  - {pred['file']}")
            fp_count += 1
    if fp_count == 0:
        print("  None!")
    
    print("\nMisclassified Spam Emails (False Negatives):")
    fn_count = 0
    for pred in spam_predictions:
        if not pred['correct'] and fn_count < 5:
            print(f"  - {pred['file']}")
            fn_count += 1
    if fn_count == 0:
        print("  None!")


if __name__ == "__main__":
    main()

