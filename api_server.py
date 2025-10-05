# api_server.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bioengine")

# Config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "chunks_faiss.index"
METADATA_FILE = "chunks_metadata.json"
DATASET_FILE = "papers_metadata.json"

# Initialize FastAPI
app = FastAPI(title="Bio Engine Semantic Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
index = None
dataset = None
metadata = None

# ---------------------- MODELS ----------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text", min_length=1)
    k: int = Field(default=5, description="Number of results", ge=1, le=50)

class SearchResult(BaseModel):
    rank: int
    score: float
    title: str
    author: str
    chunk_id: str
    text: str
    pdf_path: str
    word_count: Optional[int] = None

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    message: str
    index_loaded: bool
    model_loaded: bool
    total_chunks: int

# ---------------------- STARTUP ----------------------

def load_resources_lazy():
    """Lazy load model and index only when first needed (saves memory on startup)"""
    global model, index, dataset, metadata
    
    if model is None:
        logger.info(f"Loading model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✓ Model loaded: {EMBEDDING_MODEL}")
    
    if index is None and os.path.exists(FAISS_INDEX_FILE):
        logger.info("Loading FAISS index and dataset...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"✓ Index loaded ({index.ntotal} vectors, {len(dataset)} chunks)")
    
    return model is not None and index is not None

@app.on_event("startup")
async def startup_event():
    """Quick startup check (no heavy loading to save memory)"""
    global model, index, dataset, metadata
    logger.info("🚀 Booting Bio Engine API...")
    logger.info("💡 Using lazy loading to reduce memory footprint")

    try:
        # Just check if files exist (don't load yet)
        files_exist = all([
            os.path.exists(FAISS_INDEX_FILE),
            os.path.exists(METADATA_FILE),
            os.path.exists(DATASET_FILE)
        ])
        
        if not files_exist:
            logger.warning("⚠️ Data files not found - search will not work")
            logger.warning(f"   Missing: {FAISS_INDEX_FILE}, {METADATA_FILE}, or {DATASET_FILE}")
            logger.warning("   Build locally: python semanticSearchBuilder.py")
        else:
            logger.info("✓ Data files detected (will load on first search)")
            
        logger.info("✅ API ready - resources will load on demand")

    except Exception as e:
        logger.error(f"Startup check failed: {e}")

# ---------------------- ROUTES ----------------------

@app.get("/")
async def root():
    return {
        "message": "Bio Engine Semantic Search API",
        "docs": "/docs",
        "health": "/health",
        "search": "/search",
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    ready = all([model, index, dataset])
    return {
        "status": "healthy" if ready else "unhealthy",
        "message": "Ready" if ready else "Index/model not loaded",
        "index_loaded": index is not None,
        "model_loaded": model is not None,
        "total_chunks": len(dataset) if dataset else 0,
    }

@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    # Lazy load resources on first search request
    try:
        if not load_resources_lazy():
            raise HTTPException(
                status_code=503, 
                detail="Search unavailable. Required data files not found. Please upload index files or build them locally."
            )
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to initialize search: {str(e)}")

    try:
        q_emb = model.encode([request.query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, request.k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(dataset):
                continue
            chunk = dataset[idx]
            results.append(SearchResult(
                rank=rank,
                score=float(score),
                title=chunk.get("metadata", {}).get("title", ""),
                author=chunk.get("metadata", {}).get("author", ""),
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                pdf_path=chunk.get("pdf_path", ""),
                word_count=chunk.get("word_count"),
            ))

        logger.info(f"🔍 Query: '{request.query}' → {len(results)} results")
        return SearchResponse(query=request.query, total_results=len(results), results=results)

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=SearchResponse)
async def search_get(query: str, k: int = 5):
    return await search_post(SearchRequest(query=query, k=k))

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
