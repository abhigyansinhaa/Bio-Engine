import os
import json
import fitz  # PyMuPDF for PDF parsing
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
PDF_DIR = "sample_publications"       
OUTPUT_JSON = "papers_metadata.json"
MAX_CHUNK_TOKENS = 2000           # max number of words per chunk for LLM
OVERLAP_TOKENS = 200              # overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"  # Separate metadata for quick lookup
# ----------------------------------------

def clean_text(text):
    """Enhanced text cleanup for better processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and common PDF artifacts
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)  # Page numbers at line end
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)      # Standalone page numbers
    
    # Clean up common PDF extraction issues
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars that might be artifacts
    text = re.sub(r'\b\w{1}\b', '', text)       # Remove single characters (often artifacts)
    
    # Remove URLs and email addresses (often noisy in academic papers)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    return text.strip()

def extract_metadata_and_text(pdf_path):
    """Extract metadata (title, author) + text from PDF with better error handling."""
    metadata = {}
    text = ""
    
    try:
        with fitz.open(pdf_path) as doc:
            # Extract metadata
            meta = doc.metadata
            filename = Path(pdf_path).stem
            
            # Try to get title from metadata, fallback to filename
            title = meta.get("title", "").strip()
            if not title or len(title) < 3:
                # Use filename without extension as title
                title = filename.replace('_', ' ').replace('-', ' ')
                # Clean up the filename-based title
                title = re.sub(r'\d{3}_', '', title)  # Remove leading numbers like "001_"
                title = title.title()  # Convert to title case
            
            metadata = {
                "title": title,
                "author": meta.get("author", "Unknown").strip() or "Unknown",
                "subject": meta.get("subject", "").strip(),
                "keywords": meta.get("keywords", "").strip(),
                "creator": meta.get("creator", "").strip(),
                "producer": meta.get("producer", "").strip(),
                "creation_date": str(meta.get("creationDate", "")),
                "modification_date": str(meta.get("modDate", "")),
                "page_count": len(doc),
                "filename": filename
            }
            
            # Extract text from all pages
            page_texts = []
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():  # Only add non-empty pages
                        page_texts.append(clean_text(page_text))
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
            
            text = "\n\n".join(page_texts)
            
    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")
        # Create minimal metadata for failed PDFs
        metadata = {
            "title": Path(pdf_path).stem,
            "author": "Unknown",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "page_count": 0,
            "filename": Path(pdf_path).stem,
            "parsing_error": str(e)
        }
    
    return metadata, text

def chunk_text(text, max_tokens=2000, overlap_tokens=200):
    """Split text into overlapping chunks optimized for LLM processing."""
    if not text or not text.strip():
        return []
    
    # Split by sentences first for better chunk boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        # If adding this sentence would exceed max tokens, finalize current chunk
        if current_token_count + sentence_tokens > max_tokens and current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Start new chunk with overlap from previous chunk
            overlap_sentences = []
            overlap_count = 0
            
            # Go backwards through current chunk to build overlap
            for i in range(len(current_chunk) - 1, -1, -1):
                sentence_token_count = len(current_chunk[i].split())
                if overlap_count + sentence_token_count <= overlap_tokens:
                    overlap_sentences.insert(0, current_chunk[i])
                    overlap_count += sentence_token_count
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_token_count = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    # Filter out very short chunks (likely noise) - increased threshold for larger chunks
    chunks = [chunk for chunk in chunks if len(chunk.split()) >= 50]
    
    return chunks

def build_json_dataset(pdf_dir, output_file):
    """Extract metadata + chunks from PDFs into JSON with progress tracking."""
    dataset = []
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return []
    
    logger.info(f"Processing {len(pdf_files)} PDF files...")
    
    total_chunks = 0
    processed_files = 0
    failed_files = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            metadata, text = extract_metadata_and_text(pdf_file)
            
            if not text or len(text.strip()) < 100:  # Skip very short or empty texts
                logger.warning(f"Skipping {pdf_file.name}: insufficient text content")
                failed_files += 1
                continue
            
            chunks = chunk_text(text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)
            
            if not chunks:
                logger.warning(f"No valid chunks created for {pdf_file.name}")
                failed_files += 1
                continue
            
            file_chunks = 0
            for i, chunk in enumerate(chunks):
                dataset.append({
                    "pdf_path": str(pdf_file),
                    "chunk_id": f"{pdf_file.stem}_chunk_{i}",
                    "chunk_index": i,
                    "text": chunk,
                    "metadata": metadata,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk)
                })
                file_chunks += 1
            
            total_chunks += file_chunks
            processed_files += 1
            logger.info(f"✓ {pdf_file.name}: {file_chunks} chunks created")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            failed_files += 1
    
    # Save the dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Save metadata separately for quick lookup
    metadata_only = []
    for item in dataset:
        metadata_only.append({
            "chunk_id": item["chunk_id"],
            "pdf_path": item["pdf_path"],
            "chunk_index": item["chunk_index"],
            "metadata": item["metadata"],
            "word_count": item["word_count"]
        })
    
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_only, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Dataset created:")
    logger.info(f"   - Files processed: {processed_files}")
    logger.info(f"   - Files failed: {failed_files}")
    logger.info(f"   - Total chunks: {total_chunks}")
    logger.info(f"   - Avg chunks per file: {total_chunks/max(processed_files, 1):.1f}")
    logger.info(f"   - Saved to: {output_file}")
    
    return dataset

def build_faiss_index(new_dataset,model_name=EMBEDDING_MODEL,index_file=FAISS_INDEX_FILE,metadata_file=METADATA_FILE):
    """
    Build FAISS index with new dataset chunks.
    """
    logger.info("Building FAISS index...")

    # Load existing index + metadata if they exist
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        logger.info("Loading existing FAISS index and metadata...")
        index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.info("No existing index found. Creating new one...")
        index = None
        metadata = []

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Prepare new embeddings
    texts = [item["text"] for item in new_dataset]
    logger.info(f"Generating embeddings for {len(texts)} new chunks...")
    new_embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        batch_size=32
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(new_embeddings)

    # If no existing index → initialize
    if index is None:
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

    # Add new embeddings
    index.add(new_embeddings)
    logger.info(f"Added {len(new_embeddings)} new vectors. Total: {index.ntotal}")

    # Append metadata
    for item in new_dataset:
        metadata.append({
            "chunk_id": item["chunk_id"],
            "pdf_path": item["pdf_path"],
            "chunk_index": item["chunk_index"],
            "metadata": item["metadata"],
            "word_count": item["word_count"]
        })

    # Save updated index + metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("✅ FAISS index built and saved.")
    return index, metadata


def load_and_search_example(query="machine learning applications", k=5, pretty=True):
    """Demonstrate semantic search on the FAISS index."""
    try:
        # Load the index and metadata
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
            logger.error("Index or metadata files not found. Run main() first.")
            return
        
        index = faiss.read_index(FAISS_INDEX_FILE)
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Load model and encode query
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, k)
        
        if pretty:
            print("\n" + "="*80)
            print(f" 🔎 Top {k} results for query: '{query}'")
            print("="*80)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = dataset[idx]
            
            result = {
                "rank": i + 1,
                "score": float(score),
                "title": chunk['metadata']['title'],
                "author": chunk['metadata']['author'],
                "chunk_id": chunk['chunk_id'],
                "text": chunk['text']
            }
            results.append(result)
            
            if pretty:
                print(f"\n{i+1}. Score: {score:.4f}")
                print(f"   📄 Title: {chunk['metadata']['title']}")
                print(f"   👤 Author: {chunk['metadata']['author']}")
                print(f"   📑 Chunk ID: {chunk['chunk_id']}")
                print(f"   📝 Preview: {chunk['text'][:300]}...")
                print("-"*80)
            else:
                logger.info(f"\n{i+1}. Score: {score:.4f}")
                logger.info(f"   Title: {chunk['metadata']['title']}")
                logger.info(f"   Author: {chunk['metadata']['author']}")
                logger.info(f"   Chunk: {chunk['chunk_id']}")
                logger.info(f"   Text: {chunk['text'][:200]}...")
        
        return results
    
    except Exception as e:
        logger.error(f"Search example failed: {e}")
        return []

def main():
    """Main execution function."""
    logger.info("Starting PDF processing and FAISS indexing...")
    
    # Build dataset from PDFs
    dataset = build_json_dataset(PDF_DIR, OUTPUT_JSON)
    
    if not dataset:
        logger.error("No dataset created. Check PDF directory and files.")
        return
    
    # Build FAISS index
    index, embeddings = build_faiss_index(dataset)
    
    if index is None:
        logger.error("Failed to build FAISS index.")
        return
    
    logger.info("\n✅ Processing complete! You can now:")
    logger.info("1. Use the FAISS index for semantic search")
    logger.info("2. Load the JSON dataset for analysis")
    logger.info("3. Run load_and_search_example() to test search")
    
    # Run a quick test search
    if input("\nRun a test search? (y/n): ").lower().startswith('y'):
        test_query = input("Enter search query (or press Enter for default): ").strip()
        if not test_query:
            test_query = "machine learning applications"
        load_and_search_example(test_query)

if __name__ == "__main__":
    main()