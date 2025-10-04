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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - matches semanticSearchBuilder.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "chunks_faiss.index"
METADATA_FILE = "chunks_metadata.json"
DATASET_FILE = "papers_metadata.json"

# Initialize FastAPI app
app = FastAPI(
    title="Bio Engine Semantic Search API",
    description="API for searching biomedical publications using semantic search",
    version="1.0.0"
)

# Configure CORS (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded resources
model = None
index = None
dataset = None
metadata = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text", min_length=1)
    k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "microgravity bone density loss",
                "k": 5
            }
        }

class SearchResult(BaseModel):
    rank: int
    score: float
    title: str
    author: str
    chunk_id: str
    text: str
    pdf_path: str
    word_count: Optional[int] = None
    chunk_index: Optional[int] = None

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

@app.on_event("startup")
async def startup_event():
    """Load model and index on startup"""
    global model, index, dataset, metadata
    
    try:
        logger.info("Loading FAISS index and model...")
        
        # Check if required files exist
        if not os.path.exists(FAISS_INDEX_FILE):
            logger.error(f"FAISS index not found: {FAISS_INDEX_FILE}")
            logger.error("Please run: python semanticSearchBuilder.py")
            return
        
        if not os.path.exists(METADATA_FILE):
            logger.error(f"Metadata file not found: {METADATA_FILE}")
            return
        
        if not os.path.exists(DATASET_FILE):
            logger.error(f"Dataset file not found: {DATASET_FILE}")
            return
        
        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_FILE)
        logger.info(f"✓ FAISS index loaded: {index.ntotal} vectors")
        
        # Load metadata
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"✓ Metadata loaded: {len(metadata)} chunks")
        
        # Load full dataset (contains text)
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"✓ Dataset loaded: {len(dataset)} chunks")
        
        # Load sentence transformer model
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")
        
        logger.info("🚀 Server ready!")
        
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        logger.error("Server will start but search functionality will not work")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bio Engine Semantic Search API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search_post": "/search (POST)",
            "search_get": "/search (GET)",
            "stats": "/stats"
        },
        "documentation": "/docs"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy" if (model and index and dataset) else "unhealthy",
        "message": "All systems operational" if (model and index and dataset) else "System not ready",
        "index_loaded": index is not None,
        "model_loaded": model is not None,
        "total_chunks": len(dataset) if dataset else 0
    }

# Search endpoint (POST method)
@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    Search for relevant chunks using semantic search (POST method)
    
    - **query**: The search query text
    - **k**: Number of top results to return (default: 5, max: 50)
    """
    if not all([model, index, dataset]):
        raise HTTPException(
            status_code=503,
            detail="Search service not available. Index or model not loaded. Please run: python semanticSearchBuilder.py"
        )
    
    try:
        # Encode query
        query_embedding = model.encode([request.query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = index.search(query_embedding, request.k)
        
        # Build results with full text
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(dataset):
                continue
                
            chunk = dataset[idx]
            
            result = SearchResult(
                rank=i + 1,
                score=float(score),
                title=chunk.get('metadata', {}).get('title', ''),
                author=chunk.get('metadata', {}).get('author', ''),
                chunk_id=chunk.get('chunk_id', ''),
                text=chunk.get('text', ''),
                pdf_path=chunk.get('pdf_path', ''),
                word_count=chunk.get('word_count'),
                chunk_index=chunk.get('chunk_index')
            )
            results.append(result)
        
        logger.info(f"Search query: '{request.query}' - {len(results)} results")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Search endpoint (GET method)
@app.get("/search", response_model=SearchResponse)
async def search_get(
    query: str = Query(..., description="Search query text", min_length=1),
    k: int = Query(5, description="Number of results to return", ge=1, le=50)
):
    """
    Search for relevant chunks using semantic search (GET method for browser testing)
    
    - **query**: The search query text
    - **k**: Number of top results to return (default: 5, max: 50)
    """
    return await search_post(SearchRequest(query=query, k=k))

# Statistics endpoint
@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed papers"""
    if not dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    try:
        # Calculate statistics
        unique_papers = set()
        unique_titles = set()
        total_words = 0
        total_chunks = len(dataset)
        
        for chunk in dataset:
            metadata_obj = chunk.get('metadata', {})
            if 'filename' in metadata_obj:
                unique_papers.add(metadata_obj['filename'])
            if 'title' in metadata_obj:
                unique_titles.add(metadata_obj['title'])
            total_words += chunk.get('word_count', 0)
        
        return {
            "total_papers": len(unique_papers) or len(unique_titles),
            "total_chunks": total_chunks,
            "total_words": total_words,
            "avg_chunks_per_paper": round(total_chunks / max(len(unique_papers), len(unique_titles), 1), 2),
            "avg_words_per_chunk": round(total_words / total_chunks, 2) if total_chunks else 0,
            "embedding_model": EMBEDDING_MODEL,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
            "embedding_dimension": index.d if index else None
        }
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Check if required files exist before starting
    files_exist = all([
        os.path.exists(FAISS_INDEX_FILE),
        os.path.exists(METADATA_FILE),
        os.path.exists(DATASET_FILE)
    ])
    
    if not files_exist:
        logger.error("❌ Required files not found!")
        logger.error("Please build the index first by running:")
        logger.error("  python semanticSearchBuilder.py")
        logger.error("\nRequired files:")
        logger.error(f"  - {FAISS_INDEX_FILE}")
        logger.error(f"  - {METADATA_FILE}")
        logger.error(f"  - {DATASET_FILE}")
        logger.error("\nStarting server anyway (search will not work until files are generated)...")
    
    logger.info("🚀 Starting Bio Engine API Server...")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Alternative Docs: http://localhost:8000/redoc")
    logger.info("💻 Server: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
