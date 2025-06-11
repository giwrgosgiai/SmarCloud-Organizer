import csv
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import os

class FeedbackTrainer:
    def __init__(self, feedback_file: str = 'feedback_log.csv'):
        self.feedback_file = feedback_file
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load rules from feedback log"""
        rules = defaultdict(lambda: {
            'move': {'count': 0, 'patterns': set()},
            'associate': {'count': 0, 'patterns': set()},
            'ignore': {'count': 0, 'patterns': set()}
        })

        if not os.path.exists(self.feedback_file):
            return rules

        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) >= 4:
                        filename, doc_type, action, explanation = row

                        # Extract patterns from filename
                        patterns = self._extract_patterns(filename)

                        # Update rules
                        if doc_type in rules:
                            if action in rules[doc_type]:
                                rules[doc_type][action]['count'] += 1
                                rules[doc_type][action]['patterns'].update(patterns)
        except Exception as e:
            print(f"Error loading feedback rules: {str(e)}")

        return rules

    def _extract_patterns(self, filename: str) -> set:
        """Extract patterns from filename"""
        patterns = set()

        # Add whole filename as pattern
        patterns.add(filename.lower())

        # Add words as patterns
        words = re.findall(r'\w+', filename.lower())
        patterns.update(words)

        # Add common patterns
        patterns.add(re.sub(r'\d+', 'NUM', filename.lower()))
        patterns.add(re.sub(r'[^a-z]', '', filename.lower()))

        return patterns

    def suggest_action(self, filename: str, doc_type: str) -> Tuple[str, List[str]]:
        """
        Suggest an action for a file based on learned rules.

        Args:
            filename: The filename to analyze
            doc_type: The detected document type

        Returns:
            Tuple of (suggested_action, list_of_explanations)
        """
        if doc_type not in self.rules:
            return 'ignore', ['No rules found for this document type']

        # Extract patterns from the filename
        file_patterns = self._extract_patterns(filename)

        # Calculate scores for each action
        scores = {
            'move': 0,
            'associate': 0,
            'ignore': 0
        }

        explanations = []

        for action, data in self.rules[doc_type].items():
            # Base score on action count
            scores[action] = data['count']

            # Add score for pattern matches
            for pattern in file_patterns:
                if any(pattern in rule_pattern for rule_pattern in data['patterns']):
                    scores[action] += 1

            explanations.append(f"Action '{action}' has score {scores[action]} based on {data['count']} previous actions")

        # Get the action with highest score
        best_action = max(scores.items(), key=lambda x: x[1])[0]

        # If all scores are 0, default to ignore
        if scores[best_action] == 0:
            best_action = 'ignore'
            explanations.append("No matching patterns found, defaulting to ignore")

        return best_action, explanations

    def add_feedback(self, filename: str, doc_type: str, action: str, explanation: str) -> None:
        """Add new feedback to the log"""
        try:
            # Create file with header if it doesn't exist
            if not os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['filename', 'document_type', 'action', 'explanation'])

            # Append new feedback
            with open(self.feedback_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, doc_type, action, explanation])

            # Reload rules
            self.rules = self._load_rules()

        except Exception as e:
            print(f"Error adding feedback: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learned rules"""
        stats = {
            'total_feedback': 0,
            'actions_by_type': defaultdict(lambda: defaultdict(int)),
            'most_common_patterns': defaultdict(int)
        }

        if not os.path.exists(self.feedback_file):
            return stats

        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) >= 4:
                        filename, doc_type, action, _ = row
                        stats['total_feedback'] += 1
                        stats['actions_by_type'][doc_type][action] += 1

                        # Count patterns
                        for pattern in self._extract_patterns(filename):
                            stats['most_common_patterns'][pattern] += 1
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")

        return stats

if __name__ == '__main__':
    trainer = FeedbackTrainer()
    # Example usage
    test_cases = [
        ('translated_invoice.pdf', 'invoice'),
        ('manual_guide.pdf', 'manual'),
        ('container2_invoice.pdf', 'invoice'),
        ('random_file.pdf', 'other'),
    ]
    for fname, dtype in test_cases:
        action, explanation = trainer.suggest_action(fname, dtype)
        print(f"File: {fname}, DocType: {dtype} → Suggested: {action}")
        print(f"  Explanation: {'; '.join(explanation)}\n")