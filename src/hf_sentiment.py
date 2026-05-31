"""Local sentiment polarity via a Hugging Face text-classification model.

Uses a social-media-tuned RoBERTa (see config.SENTIMENT_HF_MODEL) with a native
3-way head: negative / neutral / positive. For each input we return:
  - sentiment = argmax label
  - score     = p_pos - p_neg   (signed polarity in [-1, +1])
  - p_pos     = probability of the positive class

The model + tokenizer are loaded lazily on first use so importing this module is cheap.
"""

from __future__ import annotations

from config import SENTIMENT_HF_MODEL

_pipeline = None


def _get_pipeline():
    """Lazy-load the text-classification pipeline (downloads weights on first call)."""
    global _pipeline
    if _pipeline is None:
        # Imported here so the heavy torch/transformers import only happens when needed.
        from transformers import pipeline

        _pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_HF_MODEL,
            top_k=None,          # return scores for ALL classes
            truncation=True,
            max_length=512,
        )
    return _pipeline


# Map possible label spellings to our canonical 3-way vocabulary.
_LABEL_MAP = {
    "negative": "negative", "neg": "negative", "label_0": "negative",
    "neutral": "neutral", "neu": "neutral", "label_1": "neutral",
    "positive": "positive", "pos": "positive", "label_2": "positive",
}


def _canon(label: str) -> str:
    return _LABEL_MAP.get(label.strip().lower(), label.strip().lower())


def classify(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Classify a list of texts. Returns one dict per input:
    {"sentiment": "...", "score": float (-1..1), "p_pos": float}.
    """
    if not texts:
        return []
    pipe = _get_pipeline()
    raw = pipe(texts, batch_size=batch_size)

    out: list[dict] = []
    for item in raw:
        # item is a list of {"label": <class>, "score": <prob>} for each class.
        probs = {_canon(c.get("label", "")): float(c.get("score", 0.0)) for c in item}
        p_pos = probs.get("positive", 0.0)
        p_neg = probs.get("negative", 0.0)
        sentiment = max(probs, key=probs.get) if probs else "neutral"
        out.append({
            "sentiment": sentiment,
            "score": round(p_pos - p_neg, 4),
            "p_pos": round(p_pos, 4),
        })
    return out
