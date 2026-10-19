import logging, math, os, re, threading, time
# Lifespan auto-reload trigger
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.content import PAPER

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE.parent / "model"))
MAX_LEN = 128  # sama dengan saat pelatihan
LABELS = ["Tidak Terindikasi", "Terindikasi Depresi"]  # 0, 1 (sesuai LABEL_MAP notebook)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("depresi")
state: dict = {"ready": False, "error": None}
lock = threading.Lock()


def _predict(text: str) -> dict:
    tok, bert, s2 = state["tok"], state["bert"], state["s2"]
    enc = tok([text], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="tf")
    out = bert.bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    token_type_ids=enc["token_type_ids"], training=False)
    pooled = out.pooler_output                       # embedding [CLS] (768)
    logits = bert.classifier(pooled).numpy()[0]      # S1
    e = np.exp(logits - logits.max()); p1 = float((e / e.sum())[1])
    z = (pooled.numpy()[0] - s2["mean"]) / s2["scale"]  # S2
    margin = float(z @ s2["coef"].ravel() + s2["intercept"].ravel()[0])
    p2 = 1.0 / (1.0 + math.exp(-margin))             # skor relatif, bukan probabilitas terkalibrasi
    return {"s1": p1, "s2": p2}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        t0 = time.time()
        log.info("Memuat model dari %s", MODEL_DIR)
        from transformers import BertTokenizer, TFBertForSequenceClassification
        state["tok"] = BertTokenizer.from_pretrained(str(MODEL_DIR / "s1"))
        state["bert"] = TFBertForSequenceClassification.from_pretrained(str(MODEL_DIR / "s1"))
        p = np.load(MODEL_DIR / "s2_params.npz")
        state["s2"] = {k: p[k] for k in ("coef", "intercept", "mean", "scale")}
        _predict("pemanasan model")
        state["ready"] = True
        log.info("Model siap dalam %.1f detik", time.time() - t0)
    except Exception as exc:  # tampilkan di Container logs
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("GAGAL memuat model. Pastikan folder model/s1 dan model/s2_params.npz ada.")
    yield


app = FastAPI(title="DepresiKU · Deteksi Dini Depresi", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


class PredictIn(BaseModel):
    text: str = Field(min_length=3, max_length=1000)
    mode: Literal["s1", "s2", "both"] = "both"


@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse(request, "landing.html", {"p": PAPER})


@app.get("/deteksi")
def detector(request: Request):
    return templates.TemplateResponse(request, "index.html", {"p": PAPER})


@app.get("/api/health")
def health():
    return {"status": "ok" if state["ready"] else "error", "detail": state["error"]}


@app.post("/api/predict")
def predict(body: PredictIn):
    if not state["ready"]:
        raise HTTPException(503, f"Model belum siap. {state['error'] or 'Sedang memuat...'}")
    text = re.sub(r"\s+", " ", body.text).strip()
    t0 = time.time()
    with lock:
        probs = _predict(text)
    keys = ["s1", "s2"] if body.mode == "both" else [body.mode]
    results = {k: {"label": LABELS[int(probs[k] >= 0.5)], "p_depresi": round(probs[k], 4)} for k in keys}
    return {"results": results, "agree": len({r["label"] for r in results.values()}) == 1,
            "ms": round((time.time() - t0) * 1000)}
