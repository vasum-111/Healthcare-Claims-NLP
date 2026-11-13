from fastapi.testclient import TestClient
from src.data import generate
from src.model import fit_baseline
from src.entities import extract
from src.api import app


def test_template_families_never_cross_splits():
    rows = generate()
    groups = {
        s: {r["family"] for r in rows if r["split"] == s}
        for s in ["train", "validation", "test"]
    }
    assert not groups["train"] & (groups["validation"] | groups["test"])
    assert not groups["validation"] & groups["test"]


def test_generator_is_reproducible_and_covers_labels():
    rows = generate()
    assert rows == generate()
    assert len(rows) == 432
    for split in ["train", "validation", "test"]:
        assert {r["label"] for r in rows if r["split"] == split} == {
            "clinical",
            "billing",
            "eligibility",
        }


def test_vectorizer_does_not_fit_test_text():
    rows = generate()
    rows.append({"text": "secretunseentoken", "label": "clinical", "split": "test"})
    model = fit_baseline(rows)
    assert "secretunseentoken" not in model[0].vocabulary_


def test_entities_preserve_offsets():
    text = "Claim CLM-12345 costs 60 dollars on 2026-11-12 for E11.9"
    entities = extract(text)
    assert {e["type"] for e in entities} == {
        "claim_id",
        "amount",
        "date",
        "diagnosis_code",
    }
    assert all(text[e["start"] : e["end"]] == e["value"] for e in entities)


def test_api_and_limits():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "x" * 2001}).status_code == 422
    result = client.post(
        "/classify",
        json={"text": "My invoice has a duplicate billing charge of 60 dollars"},
    ).json()
    assert result["label"] == "billing"
    assert 0 <= result["confidence"] <= 1
