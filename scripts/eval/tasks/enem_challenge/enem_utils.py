"""
Utility functions for the enem_challenge lm-eval task.
Removes anulada (invalidated) questions that have answerKey not in A-E.
"""

VALID_KEYS = {"A", "B", "C", "D", "E"}

def process_docs(dataset):
    """Filter out nullified questions from the ENEM dataset."""
    return dataset.filter(lambda doc: doc["answerKey"] in VALID_KEYS)
