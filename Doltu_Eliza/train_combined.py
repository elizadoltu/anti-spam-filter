#!/usr/bin/env python3
"""
Training script that combines multiple datasets.
Usage: python train_combined.py <output_model>
"""
import sys
from pathlib import Path
from classifier import SpamClassifier
from utils import EmailParser


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_combined.py <output_model>")
        print("Example: python train_combined.py trained_model.json")
        sys.exit(1)
    
    output_model = sys.argv[1]
    
    # Define all training folders
    training_folders = [
        ("../Lot1_/Lot1/Clean", "../Lot1_/Lot1/Spam"),
        ("../Lot2/Clean", "../Lot2/Spam"),
    ]
    
    print("Training classifier on combined datasets...")
    print("=" * 60)
    
    # Collect all emails
    all_clean_emails = []
    all_spam_emails = []
    
    for clean_folder, spam_folder in training_folders:
        clean_path = Path(clean_folder)
        spam_path = Path(spam_folder)
        
        print(f"\nLoading from:")
        print(f"  Clean: {clean_folder}")
        print(f"  Spam:  {spam_folder}")
        
        # Read clean emails
        clean_count = 0
        if clean_path.exists():
            for file_path in clean_path.iterdir():
                if file_path.is_file():
                    try:
                        email_content = EmailParser.read_email(file_path)
                        all_clean_emails.append(email_content)
                        clean_count += 1
                    except:
                        pass
        
        # Read spam emails
        spam_count = 0
        if spam_path.exists():
            for file_path in spam_path.iterdir():
                if file_path.is_file():
                    try:
                        email_content = EmailParser.read_email(file_path)
                        all_spam_emails.append(email_content)
                        spam_count += 1
                    except:
                        pass
        
        print(f"  Loaded: {clean_count} clean, {spam_count} spam")
    
    print("\n" + "=" * 60)
    print(f"Total training data:")
    print(f"  Clean emails: {len(all_clean_emails)}")
    print(f"  Spam emails:  {len(all_spam_emails)}")
    print(f"  Total:        {len(all_clean_emails) + len(all_spam_emails)}")
    print("=" * 60)
    
    # Train the classifier
    print("\nTraining Naive Bayes classifier...")
    classifier = SpamClassifier()
    
    if all_clean_emails and all_spam_emails:
        classifier.naive_bayes.train(all_clean_emails, all_spam_emails)
        classifier.trained = True
        
        # Save the model
        classifier.save_model(output_model)
        
        print(f"\n✓ Model trained and saved to: {output_model}")
        print(f"\nModel statistics:")
        print(f"  Clean documents:  {classifier.naive_bayes.clean_doc_count}")
        print(f"  Spam documents:   {classifier.naive_bayes.spam_doc_count}")
        print(f"  Vocabulary size:  {len(classifier.naive_bayes.vocabulary)}")
        print(f"  Clean words:      {classifier.naive_bayes.clean_total_words}")
        print(f"  Spam words:       {classifier.naive_bayes.spam_total_words}")
        print(f"\nClassifier parameters:")
        print(f"  NB weight:        {classifier.nb_weight}")
        print(f"  Feature weight:   {classifier.feature_weight}")
        print(f"  Feature threshold: {classifier.feature_threshold}")
    else:
        print("Error: No training data found!")
        sys.exit(1)


if __name__ == "__main__":
    main()

