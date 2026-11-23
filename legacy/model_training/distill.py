from transformers import pipeline
from tqdm import tqdm

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_sentiment_pipe = None
_emotion_pipe = None
_pipe_device = None


def _resolve_device(device):
    if device == "cuda":
        return 0
    if device == "cpu":
        return -1
    return 0 if _CUDA_AVAILABLE else -1  # auto


def _get_pipes(device=None, fp16=False):
    global _sentiment_pipe, _emotion_pipe, _pipe_device
    resolved = _resolve_device(device)
    if _sentiment_pipe is None or resolved != _pipe_device:
        _pipe_device = resolved
        label = f"cuda:{resolved}" if resolved >= 0 else "cpu"
        print(f"distill.py: loading models on {label}" + (" (fp16)" if fp16 else ""))
        kwargs = {}
        if fp16 and resolved >= 0:
            import torch
            kwargs["torch_dtype"] = torch.float16
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=resolved,
            truncation=True,
            max_length=512,
            **kwargs,
        )
        _emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=resolved,
            truncation=True,
            max_length=512,
            **kwargs,
        )
    return _sentiment_pipe, _emotion_pipe


def distill_labels(df, text_col="text", batch_size=128, device=None, fp16=False):
    """Fills in sentiment/emotion columns using the same models ai_engine.py
    already runs live - this is what turns the current pipeline into a
    teacher for the new unified model.

    device: "cuda", "cpu", or None to auto-detect. Both models are small
    (well under 100M params each) - batch_size=128 gives the GPU meaningfully
    more work per forward pass. fp16 halves memory per tensor, letting you push
    batch_size even further."""
    sentiment_pipe, emotion_pipe = _get_pipes(device=device, fp16=fp16)
    texts = df[text_col].fillna("").tolist()

    if not texts:
        df = df.copy()
        df["sentiment"] = []
        df["emotion"] = []
        return df

    sentiments, emotions = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Distilling batches", unit="batch"):
        batch = texts[i:i + batch_size]
        sentiments.extend(sentiment_pipe(batch, truncation=True, max_length=512))
        emotions.extend(emotion_pipe(batch, truncation=True, max_length=512))

    df = df.copy()
    df["sentiment"] = [s["label"].lower() for s in sentiments]
    df["emotion"] = [e["label"].lower() for e in emotions]
    return df
