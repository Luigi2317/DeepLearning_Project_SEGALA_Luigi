
"""
API FastAPI — Classification d'intentions tweets (BiLSTM)
Lancer : uvicorn api_bilstm:app --reload --port 8000
Docs   : http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pickle, json, re

app = FastAPI(title="Intent Classifier — BiLSTM", version="1.0")

VOCAB_PATH  = "../data/vocab.pkl"
LABELS_PATH = "../data/label_maps.json"
MODEL_PATH  = "../models/bilstm_optimised.pt"
MAX_LEN     = 64

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
with open(LABELS_PATH) as f:
    label_maps = json.load(f)

CLIENT_LABELS = {int(k): v for k, v in label_maps["client"].items()}
N_CLASSES     = len(CLIENT_LABELS)
LABEL_NAMES   = [CLIENT_LABELS[i] for i in range(N_CLASSES)]
PAD_IDX       = vocab["<PAD>"]


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(self.embedding(x))
        return self.fc(self.dropout(torch.cat([h[-2], h[-1]], dim=1)))


model = BiLSTM(len(vocab), 128, 256, N_CLASSES, 2, 0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def clean_tweet(text):
    """Preprocessing identique à 01_clustering.ipynb Bloc 5."""
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s?!\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    tokens = text.split()[:MAX_LEN]
    return [vocab.get(w, vocab["<UNK>"]) for w in tokens] or [vocab["<UNK>"]]


class PredictionInput(BaseModel):
    text: str

class TopPrediction(BaseModel):
    intention: str
    confiance: float

class PredictionResponse(BaseModel):
    intention: str
    confiance: float
    top3: List[TopPrediction]


@app.get("/health")
def health():
    return {"status": "ok", "modele": "BiLSTM", "classes": N_CLASSES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    token_ids = tokenize(clean_tweet(payload.text))
    tensor    = pad_sequence(
        [torch.tensor(token_ids, dtype=torch.long)],
        batch_first=True, padding_value=PAD_IDX
    )
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_val, top_idx = torch.topk(probs, k=3)
    top3 = [TopPrediction(intention=LABEL_NAMES[i], confiance=round(v.item(), 4))
            for i, v in zip(top_idx, top_val)]
    return PredictionResponse(
        intention=top3[0].intention,
        confiance=top3[0].confiance,
        top3=top3
    )
