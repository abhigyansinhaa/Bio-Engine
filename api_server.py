# api_server.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import psutil
import gc
import requests

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

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for {request.method} {request.url}")
    logger.error(f"Error details: {exc.errors()}")
    try:
        body = await request.body()
        logger.error(f"Request body: {body.decode()}")
    except:
        logger.error("Could not read request body")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None,
            "message": "Request validation failed. Check your JSON format.",
            "example": {
                "query": "your search text here",
                "k": 5
            }
        }
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

# ---------------------- FILE DOWNLOAD ----------------------

def download_file_from_url(url: str, destination: str) -> bool:
    """Download file from URL if it doesn't exist locally"""
    if not url:
        logger.info(f"No URL provided for {destination}, skipping download")
        return False
        
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}")
        return True
    
    try:
        logger.info(f"Downloading {destination} from {url}...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                # Log progress every 10MB
                if downloaded % (10 * 1024 * 1024) == 0:
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)")
                    else:
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB")
        
        logger.info(f"Successfully downloaded {destination} ({downloaded / (1024*1024):.1f}MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {destination}: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # Clean up partial download
        return False

# ---------------------- STARTUP ----------------------

def load_resources_lazy():
    """Lazy load model and index only when first needed"""
    global model, index, dataset, metadata
    
    if model is None:
        logger.info(f"Loading model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Model loaded: {EMBEDDING_MODEL}")
    
    if index is None and os.path.exists(FAISS_INDEX_FILE):
        logger.info("Loading FAISS index...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        logger.info(f"Index loaded ({index.ntotal} vectors)")
        
    if metadata is None and os.path.exists(METADATA_FILE):
        logger.info("Loading metadata...")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded ({len(metadata)} entries)")
        
    if dataset is None and os.path.exists(DATASET_FILE):
        logger.info("Loading dataset...")
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Dataset loaded ({len(dataset)} chunks)")
    
    return model is not None and index is not None

@app.on_event("startup")
async def startup_event():
    """Startup with file download capability"""
    global model, index, dataset, metadata
    
    logger.info("🚀 Booting Bio Engine API...")
    
    # Download data files from cloud storage if URLs are provided
    # Set these as environment variables in Railway
    data_files = {
        FAISS_INDEX_FILE: os.getenv("FAISS_INDEX_URL", ""),
        METADATA_FILE: os.getenv("METADATA_URL", ""),
        DATASET_FILE: os.getenv("DATASET_URL", "")
    }
    
    logger.info("Checking for remote data files...")
    for filename, url in data_files.items():
        if url:
            download_file_from_url(url, filename)
        elif not os.path.exists(filename):
            logger.warning(f"File not found and no URL provided: {filename}")
    
    # Check which files are available
    files_exist = {
        "index": os.path.exists(FAISS_INDEX_FILE),
        "metadata": os.path.exists(METADATA_FILE),
        "dataset": os.path.exists(DATASET_FILE)
    }
    
    logger.info(f"Files available: {files_exist}")
    
    if not all(files_exist.values()):
        logger.warning("⚠️  Some data files not found - search may not work properly")
        logger.warning("   Upload files to cloud storage and set environment variables:")
        logger.warning("   - FAISS_INDEX_URL")
        logger.warning("   - METADATA_URL")
        logger.warning("   - DATASET_URL")
    else:
        logger.info("✓ All data files available (will load on first search)")
    
    logger.info("✅ API ready - resources will load on demand")

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
    logger.info(f"Search request: query='{request.query}', k={request.k}")
    
    # Lazy load resources on first search
    try:
        if not load_resources_lazy():
            raise HTTPException(
                status_code=503,
                detail="Search unavailable. Required data files not found. Set FAISS_INDEX_URL, METADATA_URL, and DATASET_URL environment variables."
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

        logger.info(f"Query '{request.query}' returned {len(results)} results")
        return SearchResponse(query=request.query, total_results=len(results), results=results)

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=SearchResponse)
async def search_get(query: str, k: int = 5):
    return await search_post(SearchRequest(query=query, k=k))

# ---------------------- MEMORY MONITORING ----------------------

@app.get("/memory")
async def get_memory_stats():
    """Get current memory usage statistics"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        system_memory = psutil.virtual_memory()
        
        return {
            "process": {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(memory_percent, 2),
            },
            "system": {
                "total_mb": round(system_memory.total / 1024 / 1024, 2),
                "available_mb": round(system_memory.available / 1024 / 1024, 2),
                "percent": round(system_memory.percent, 2),
            },
            "resources_loaded": {
                "model": model is not None,
                "index": index is not None,
                "dataset": dataset is not None,
            },
            "status": "healthy" if memory_percent < 80 else "warning" if memory_percent < 95 else "critical"
        }
    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        return {"error": str(e)}

@app.post("/gc")
async def force_garbage_collection():
    """Force garbage collection"""
    try:
        before = psutil.Process().memory_info().rss / 1024 / 1024
        collected = gc.collect()
        after = psutil.Process().memory_info().rss / 1024 / 1024
        freed = before - after
        
        return {
            "collected_objects": collected,
            "memory_before_mb": round(before, 2),
            "memory_after_mb": round(after, 2),
            "memory_freed_mb": round(freed, 2),
            "message": f"Freed {round(freed, 2)} MB"
        }
    except Exception as e:
        logger.error(f"GC error: {e}")
        return {"error": str(e)}

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)