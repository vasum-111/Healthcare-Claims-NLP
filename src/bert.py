"""Small BERT trained locally from scratch; intentionally no pretrained claim."""

import re, json, random
import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification
from .data import ROOT

LABELS = ["billing", "clinical", "eligibility"]


def tokenize(text):
    return re.findall(r"[a-z]+|\d+|[^\w\s]", text.lower())


def encode(texts, vocab, length=64):
    ids = []
    masks = []
    for text in texts:
        seq = [2] + [vocab.get(t, 1) for t in tokenize(text)][: length - 2] + [3]
        masks.append([1] * len(seq) + [0] * (length - len(seq)))
        ids.append(seq + [0] * (length - len(seq)))
    return torch.tensor(ids), torch.tensor(masks)


def fit_bert(rows, epochs=35):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for token in sorted({t for r in train for t in tokenize(r["text"])}):
        vocab[token] = len(vocab)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={v: i for i, v in enumerate(LABELS)},
    )
    model = BertForSequenceClassification(config)
    x, mask = encode([r["text"] for r in train], vocab)
    y = torch.tensor([LABELS.index(r["label"]) for r in train])
    vx, vm = encode([r["text"] for r in val], vocab)
    vy = torch.tensor([LABELS.index(r["label"]) for r in val])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    best_loss = float("inf")
    best = None
    history = []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x)).split(32):
            opt.zero_grad()
            loss = model(input_ids=x[idx], attention_mask=mask[idx], labels=y[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        model.eval()
        with torch.no_grad():
            validation = float(model(input_ids=vx, attention_mask=vm, labels=vy).loss)
        history.append({"epoch": epoch + 1, "validation_loss": validation})
        if validation < best_loss:
            best_loss = validation
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    artifacts = ROOT / "artifacts/bert"
    artifacts.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifacts)
    (artifacts / "vocab.json").write_text(json.dumps(vocab))

    def predict(texts):
        px, pm = encode(texts, vocab)
        with torch.no_grad():
            logits = model(input_ids=px, attention_mask=pm).logits
        return [LABELS[i] for i in logits.argmax(1).tolist()]

    return predict, {
        "epochs": epochs,
        "selected_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(vocab),
        "initialization": "random, no pretrained weights",
        "tokenizer": "train-only lowercase regex vocabulary",
        "history": history,
    }
