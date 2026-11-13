import argparse, json, hashlib, itertools
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from .data import load, ROOT
from .model import fit_baseline


def cluster_interval(rows, predictions):
    # Resample whole template families, not correlated message variants.
    rng = np.random.default_rng(42)
    samples = []
    groups_by_label = {
        label: sorted({r["family"] for r in rows if r["label"] == label})
        for label in ["billing", "clinical", "eligibility"]
    }
    for _ in range(600):
        chosen = [
            group
            for groups in groups_by_label.values()
            for group in rng.choice(groups, len(groups), replace=True)
        ]
        idx = [i for g in chosen for i, r in enumerate(rows) if r["family"] == g]
        samples.append(
            f1_score(
                [rows[i]["label"] for i in idx],
                [predictions[i] for i in idx],
                labels=["billing", "clinical", "eligibility"],
                average="macro",
                zero_division=0,
            )
        )
    return list(map(float, np.percentile(samples, [2.5, 97.5])))


def evaluate(include_bert=False, epochs=35):
    rows = load()
    test = [r for r in rows if r["split"] == "test"]
    truth = [r["label"] for r in test]
    baseline = fit_baseline(rows)
    pred = baseline.predict([r["text"] for r in test]).tolist()

    def score(prediction):
        return {
            "classification_report": classification_report(
                truth, prediction, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                truth, prediction, labels=["billing", "clinical", "eligibility"]
            ).tolist(),
            "macro_f1_family_bootstrap_95ci": cluster_interval(test, prediction),
        }

    report = {
        "dataset": "synthetic templates; small correlated sample; not clinical validation",
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data/messages.csv").read_bytes()
        ).hexdigest(),
        "counts": {
            s: sum(r["split"] == s for r in rows)
            for s in ["train", "validation", "test"]
        },
        "split": "template families disjoint: 15 train / 3 validation / 6 test",
        "label_order": ["billing", "clinical", "eligibility"],
        "baseline": score(pred),
        "bert": None,
    }
    if include_bert:
        from .bert import fit_bert

        predict, config = fit_bert(rows, epochs)
        bp = predict([r["text"] for r in test])
        report["bert"] = {**score(bp), "configuration": config}
        only_base = sum(a == y and b != y for a, b, y in zip(pred, bp, truth))
        only_bert = sum(a != y and b == y for a, b, y in zip(pred, bp, truth))
        report["paired_comparison"] = {
            "baseline_only_correct": only_base,
            "bert_only_correct": only_bert,
            "mcnemar_exact_p": float(
                binomtest(only_base, only_base + only_bert, 0.5).pvalue
            )
            if only_base + only_bert
            else 1.0,
            "caution": "Message-level McNemar assumes independence; template variants violate it. Descriptive only; no clinical superiority claim.",
        }
        family_deltas = []
        for family in sorted({r["family"] for r in test}):
            idx = [i for i, r in enumerate(test) if r["family"] == family]
            family_deltas.append(
                float(
                    np.mean(
                        [int(bp[i] == truth[i]) - int(pred[i] == truth[i]) for i in idx]
                    )
                )
            )
        observed = abs(float(np.mean(family_deltas)))
        permutations = [
            abs(float(np.mean(np.array(family_deltas) * signs)))
            for signs in itertools.product([-1, 1], repeat=len(family_deltas))
        ]
        report["paired_comparison"]["family_sign_permutation_p"] = float(
            np.mean(np.array(permutations) >= observed - 1e-12)
        )
        report["paired_comparison"]["family_accuracy_deltas_bert_minus_baseline"] = (
            family_deltas
        )
        report["paired_comparison"]["family_test_caution"] = (
            "Only six held-out template families; low power and synthetic selection bias."
        )
    else:
        bp = [None] * len(test)
    report["predictions"] = [
        {**r, "baseline_prediction": p, "bert_prediction": b}
        for r, p, b in zip(test, pred, bp)
    ]
    output = ROOT / "reports/evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bert", action="store_true")
    p.add_argument("--epochs", type=int, default=35)
    args = p.parse_args()
    evaluate(args.bert, args.epochs)
