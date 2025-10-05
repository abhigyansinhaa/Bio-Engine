# semanticSearchBuilder.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import gc

# ---------- CONFIG ----------
INPUT_JSON = "papers_metadata.json"
FAISS_INDEX_FILE = "chunks_faiss.index"
EMBEDDINGS_FILE = "chunks_embeddings.npy"
METADATA_FILE = "chunks_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
USE_GPU = torch.cuda.is_available()
# ----------------------------

_model_cache = None


def get_model():
    """Load the embedding model once"""
    global _model_cache
    if _model_cache is None:
        device = "cuda" if USE_GPU else "cpu"
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL} on {device}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model_cache


def load_chunks():
    """Load your JSON data"""
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"{INPUT_JSON} missing — please upload your metadata file.")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    data = load_chunks()
    n = len(data)
    model = get_model()

    print(f"[INFO] Building FAISS index for {n} chunks")
    dim = model.encode(["test"], convert_to_numpy=True).shape[1]

    index = faiss.IndexFlatIP(dim)
    metadata = []

    texts = [d.get("text", "") for d in data]
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Encoding"):
        end = min(start + BATCH_SIZE, n)
        batch = texts[start:end]
        batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(batch_emb)
        index.add(batch_emb)
        metadata.extend(data[start:end])

        # Free GPU/CPU memory
        del batch_emb
        gc.collect()
        if USE_GPU:
            torch.cuda.empty_cache()

    # Save incrementally built index
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] ✅ FAISS index built and saved to {FAISS_INDEX_FILE}")
    return index


def ensure_index_exists():
    """Called from your main app to auto-build if missing"""
    if not os.path.exists(FAISS_INDEX_FILE):
        print("[WARN] FAISS index missing — building now...")
        build_index()
    else:
        print("[INFO] Existing FAISS index found.")


if __name__ == "__main__":
    ensure_index_exists()
