from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .data import ROOT
from .model import classify
from .entities import extract

app = FastAPI(title="Claims Signal Lab", version="1.0.0")


class Message(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "dataset": "synthetic"}


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")


@app.post("/classify")
def classify_message(body: Message):
    return {
        **classify(body.text),
        "entities": extract(body.text),
        "notice": "Synthetic routing demonstration. Not for clinical or coverage decisions.",
    }
