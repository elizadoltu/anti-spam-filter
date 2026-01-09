#!/usr/bin/env python3
"""
Training script for the spam classifier.
Usage: python train.py <clean_folder> <spam_folder> [output_model]
"""
import sys
from pathlib import Path
from classifier import SpamClassifier


def main():
    if len(sys.argv) < 3:
        print("Usage: python train.py <clean_folder> <spam_folder> [output_model]")
        print("Example: python train.py data/Clean data/Spam trained_model.json")
        sys.exit(1)
    
    clean_folder = sys.argv[1]
    spam_folder = sys.argv[2]
    output_model = sys.argv[3] if len(sys.argv) > 3 else "trained_model.json"
    
    print(f"Training classifier...")
    print(f"  Clean folder: {clean_folder}")
    print(f"  Spam folder: {spam_folder}")
    
    # Create and train classifier
    classifier = SpamClassifier()
    classifier.train_from_folders(clean_folder, spam_folder)
    
    # Save the model
    classifier.save_model(output_model)
    
    print(f"\nModel trained and saved to: {output_model}")
    print(f"  Clean documents: {classifier.naive_bayes.clean_doc_count}")
    print(f"  Spam documents: {classifier.naive_bayes.spam_doc_count}")
    print(f"  Vocabulary size: {len(classifier.naive_bayes.vocabulary)}")
    print(f"  Clean words: {classifier.naive_bayes.clean_total_words}")
    print(f"  Spam words: {classifier.naive_bayes.spam_total_words}")


if __name__ == "__main__":
    main()

