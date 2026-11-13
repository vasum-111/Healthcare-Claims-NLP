"""Synthetic message templates, split by family before generating variants."""

import csv, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "clinical": [
        "My doctor prescribed {item} and I need a clinical review.",
        "Please review treatment for diagnosis {code} before my appointment.",
        "The nurse asked about my symptoms and follow-up consultation.",
        "Does my treatment plan need prior clinical authorization for {item}?",
        "I need a specialist referral to discuss my diagnosis {code}.",
        "Can a clinician review the medication requested by my physician?",
        "The hospital discharge note recommends a follow-up care visit.",
        "My physician wants medical approval for treatment {item}.",
    ],
    "billing": [
        "I was charged {amount} dollars for claim {claim}; please explain the bill.",
        "My invoice includes a duplicate charge for procedure {item}.",
        "Please explain the deductible on this statement dated {date}.",
        "The payment for claim {claim} has not appeared on my account.",
        "I want to dispute the copay amount of {amount} dollars.",
        "Why was reimbursement lower than the billed amount for {item}?",
        "The balance due on my invoice looks incorrect after payment.",
        "Please correct the billing charge for claim {claim} on {date}.",
    ],
    "eligibility": [
        "Is my membership active for the service date {date}?",
        "I need to confirm coverage after adding a dependent to my plan.",
        "When does enrollment become effective for my insurance policy?",
        "My employer changed plans; am I still eligible for benefits?",
        "Can you verify my insurance status before the visit on {date}?",
        "My member card expired; please check my coverage eligibility.",
        "Does my dependent qualify under the current policy enrollment?",
        "Please confirm the termination date for my health plan membership.",
    ],
}


def generate(seed=42, variants=18):
    rng = random.Random(seed)
    rows = []
    for label, templates in TEMPLATES.items():
        for family, template in enumerate(templates):
            split = "train" if family < 5 else "validation" if family == 5 else "test"
            for i in range(variants):
                values = dict(
                    item=rng.choice(
                        ["therapy", "procedure", "medication", "assessment"]
                    ),
                    code=rng.choice(["E11.9", "I10", "J45.909"]),
                    amount=rng.choice([25, 60, 120, 250]),
                    claim=f"CLM-{rng.randrange(10000, 99999)}",
                    date=f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
                )
                prefix = rng.choice(["Hello, ", "Please help: ", "Question: ", ""])
                rows.append(
                    {
                        "id": f"{label}-{family}-{i}",
                        "text": prefix + template.format(**values),
                        "label": label,
                        "family": f"{label}-{family}",
                        "split": split,
                    }
                )
    return rows


def save():
    rows = generate()
    path = ROOT / "data/messages.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load():
    path = ROOT / "data/messages.csv"
    if not path.exists():
        return save()
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    print(f"Wrote {len(save())} synthetic messages")
