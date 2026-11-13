"""Transparent regex entity extraction, not a clinical NER model."""

import re

PATTERNS = {
    "claim_id": r"\bCLM-\d{5}\b",
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
    "diagnosis_code": r"\b[A-Z]\d{2}(?:\.\d{1,3})?\b",
    "amount": r"\b\d+(?:\.\d{2})?\s+dollars\b",
}


def extract(text):
    return [
        {
            "type": kind,
            "value": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for kind, pattern in PATTERNS.items()
        for match in re.finditer(pattern, text)
    ]
