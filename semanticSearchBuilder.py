# build_and_query_chunks_faiss_optimized.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc
import torch

# ---------- CONFIG ----------
INPUT_JSON = "papers_metadata.json"
FAISS_INDEX_FILE = "chunks_faiss.index"
EMBEDDINGS_FILE = "chunks_embeddings.npy"
METADATA_FILE = "chunks_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512  # Increased from 256
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU
# ----------------------------

_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        device = "cuda" if USE_GPU else "cpu"
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"Model loaded on {device}!")
    return _model_cache

def load_chunks():
    print(f"Loading chunks from {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chunks")
    return data

def build_faiss_index_optimized():
    """Optimized FAISS index building with batching and GPU support"""
    dataset = load_chunks()
    n = len(dataset)
    model = get_model()

    print(f"\nProcessing {n} chunks...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Using GPU: {USE_GPU}")
    
    # Get embedding dimension
    sample_emb = model.encode(["test"], convert_to_numpy=True)
    dim = sample_emb.shape[1]
    print(f"Embedding dimension: {dim}")

    # Pre-allocate arrays
    embeddings = np.zeros((n, dim), dtype=np.float32)
    metadata = []

    # Extract texts once (avoid repeated dict lookups)
    all_texts = [d.get("text", "") for d in dataset]
    
    # Batch encoding with progress bar
    print("\nEncoding chunks in batches...")
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Encoding"):
        end = min(n, start + BATCH_SIZE)
        batch_texts = all_texts[start:end]
        
        # Encode batch (GPU accelerated if available)
        batch_emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
            normalize_embeddings=False  # We'll normalize after
        )
        embeddings[start:end] = batch_emb
        
        # Build metadata simultaneously
        for d in dataset[start:end]:
            metadata.append({
                "chunk_id": d.get("chunk_id", ""),
                "title": d.get("metadata", {}).get("title", ""),
                "author": d.get("metadata", {}).get("author", ""),
                "pdf_path": d.get("pdf_path", ""),
                "word_count": d.get("word_count", 0)
            })
        
        # Clear GPU cache periodically
        if USE_GPU and start % (BATCH_SIZE * 10) == 0:
            torch.cuda.empty_cache()

    # Normalize embeddings for cosine similarity
    print("\nNormalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    print("Building FAISS index...")
    if USE_GPU and faiss.get_num_gpus() > 0:
        # Use GPU index for faster search
        print("Building GPU-accelerated index...")
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        index = faiss.IndexFlatIP(dim)
    
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors")

    # Save everything
    print("\nSaving index and metadata...")
    
    # Convert GPU index to CPU for saving
    if USE_GPU and faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nFiles saved:")
    print(f"  Index: {FAISS_INDEX_FILE}")
    print(f"  Embeddings: {EMBEDDINGS_FILE}")
    print(f"  Metadata: {METADATA_FILE}")
    
    # Memory cleanup
    del embeddings
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    
    return index, metadata

def load_index_and_metadata():
    """Load pre-built index and metadata"""
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"Index not found. Run build_faiss_index_optimized() first.")
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def search_chunks(query, k=5):
    """Search for relevant chunks"""
    index, metadata = load_index_and_metadata()
    model = get_model()
    
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "title": m["title"],
            "author": m["author"],
            "chunk_id": m["chunk_id"],
            "pdf_path": m["pdf_path"],
            "word_count": m["word_count"]
        })
    return results

def pretty_print_search(query, k=5):
    """Pretty print search results"""
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    results = search_chunks(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Similarity Score: {r['score']:.4f}")
        print(f"   Title: {r['title']}")
        print(f"   Author: {r['author']}")
        print(f"   Chunk: {r['chunk_id']}")
        print(f"   Words: {r['word_count']}")
        print("-"*80)

def benchmark_index_building():
    """Benchmark index building performance"""
    import time
    
    print("Starting benchmark...")
    start_time = time.time()
    
    build_faiss_index_optimized()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Using GPU: {USE_GPU}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load to get chunk count
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    n_chunks = len(metadata)
    print(f"Chunks processed: {n_chunks}")
    print(f"Speed: {n_chunks/elapsed:.1f} chunks/second")
    print(f"Avg time per chunk: {elapsed/n_chunks*1000:.2f}ms")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark
        benchmark_index_building()
    elif not os.path.exists(FAISS_INDEX_FILE):
        print("Index not found. Building index...")
        build_faiss_index_optimized()
    else:
        print("Index found. Running demo search...")
        
        # Demo queries
        demo_queries = [
            "microgravity bone density loss",
            "radiation effects on DNA",
            "immune system changes in space"
        ]
        
        for query in demo_queries:
            pretty_print_search(query, k=5)
            input("\nPress Enter for next query...")