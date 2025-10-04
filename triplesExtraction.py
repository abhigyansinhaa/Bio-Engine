import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
import re

# Load API key
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ---------------- CONFIG ----------------
INPUT_JSON = "papers_metadata.json"
OUTPUT_JSON = "kg_triples.json"
CHECKPOINT_JSON = "kg_checkpoint.json"
MODEL = "gemini-2.5-flash" 
BATCH_SIZE = 10  # Save checkpoint every N chunks
DELAY_BETWEEN_CALLS = 0.5  # Rate limiting
MAX_RETRIES = 3
# ----------------------------------------

def parse_json_safely(content: str) -> List[Dict]:
    """Try to parse JSON robustly."""
    try:
        return json.loads(content)
    except:
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return []
        return []

def validate_triple(triple: Dict) -> bool:
    """Validate triple has required fields and reasonable content"""
    required_fields = ["subject", "predicate", "object"]
    
    if not all(field in triple for field in required_fields):
        return False
    
    subj = str(triple.get("subject", "")).strip()
    pred = str(triple.get("predicate", "")).strip()
    obj = str(triple.get("object", "")).strip()
    
    # Basic validation
    if not all([subj, pred, obj]):
        return False
    
    if subj.lower() == obj.lower():
        return False
    
    if any(len(x) < 2 for x in [subj, pred, obj]):
        return False
    
    if any(len(x) > 100 for x in [subj, pred, obj]):
        return False
    
    # Filter generic subjects
    if subj.lower() in ["it", "this", "that", "they", "we", "study", "research"]:
        return False
    
    return True

def normalize_triple(triple: Dict) -> Dict:
    """Normalize triple format"""
    predicate_mapping = {
        "leads to": "causes",
        "results in": "causes",
        "induces": "causes",
        "reduces": "decreases",
        "lowers": "decreases",
        "enhances": "increases",
        "inhibits": "prevents",
        "blocks": "prevents"
    }
    
    pred = triple["predicate"].strip().lower()
    normalized_pred = predicate_mapping.get(pred, pred)
    
    return {
        "subject": triple["subject"].strip().lower(),
        "predicate": normalized_pred,
        "object": triple["object"].strip().lower()
    }

def llm_extract_triples_gemini(text: str, title: str, retry_count: int = 0) -> List[Dict]:
    """Call Gemini API for triple extraction with retry logic."""
    if len(text.strip()) < 100:
        return []
    
    prompt = f"""
You are an expert biomedical knowledge extractor specializing in space biology.
Extract ONLY knowledge graph triples directly stated in the text.
Format: [{{"subject": "...", "predicate": "...", "object": "..."}}]
If no valid triples, return [].

Rules:
- Be concise, no hallucination
- Focus on: organisms, environments, biological effects, countermeasures
- Only extract explicit relationships

Publication Title: {title}

Text:
{text}

Output ONLY the JSON list.
"""
    
    try:
        model_obj = genai.GenerativeModel(MODEL)
        response = model_obj.generate_content(prompt)
        
        if not response or not response.text:
            return []
        
        raw_triples = parse_json_safely(response.text.strip())
        
        # Validate and normalize triples
        validated_triples = []
        for triple in raw_triples:
            if validate_triple(triple):
                validated_triples.append(normalize_triple(triple))
        
        return validated_triples
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"\nRetrying {title[:30]}... (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(2 ** retry_count)  # Exponential backoff
            return llm_extract_triples_gemini(text, title, retry_count + 1)
        else:
            print(f"\nFailed after {MAX_RETRIES} retries: {title[:30]}... Error: {e}")
            return []

def load_checkpoint() -> Dict:
    """Load progress from checkpoint if exists"""
    if os.path.exists(CHECKPOINT_JSON):
        try:
            with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"processed_chunks": [], "triples": []}
    return {"processed_chunks": [], "triples": []}

def save_checkpoint(checkpoint_data: Dict):
    """Save progress to checkpoint"""
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

def main():
    print("Starting knowledge graph extraction...")
    print(f"Model: {MODEL}")
    print(f"Rate limit delay: {DELAY_BETWEEN_CALLS}s")
    print("-" * 50)
    
    # Load dataset
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} chunks from {INPUT_JSON}")
    
    # Load checkpoint (resume if interrupted)
    checkpoint = load_checkpoint()
    processed_chunk_ids = set(checkpoint.get("processed_chunks", []))
    all_results = checkpoint.get("triples", [])
    
    if processed_chunk_ids:
        print(f"Resuming from checkpoint: {len(processed_chunk_ids)} chunks already processed")
    
    # Track statistics
    stats = {
        "total_chunks": len(dataset),
        "processed": len(processed_chunk_ids),
        "failed": 0,
        "total_triples": len(all_results),
        "start_time": time.time()
    }
    
    # Process chunks
    for i, chunk in enumerate(tqdm(dataset, desc="Extracting triples")):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        
        # Skip if already processed
        if chunk_id in processed_chunk_ids:
            continue
        
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Unknown")
        
        # Extract triples
        triples = llm_extract_triples_gemini(text, title)
        
        if triples:
            for t in triples:
                all_results.append({
                    "title": title,
                    "chunk_id": chunk_id,
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"]
                })
        else:
            stats["failed"] += 1
        
        # Update progress
        processed_chunk_ids.add(chunk_id)
        stats["processed"] += 1
        stats["total_triples"] = len(all_results)
        
        # Save checkpoint periodically
        if stats["processed"] % BATCH_SIZE == 0:
            checkpoint_data = {
                "processed_chunks": list(processed_chunk_ids),
                "triples": all_results,
                "stats": stats
            }
            save_checkpoint(checkpoint_data)
            
            # Progress update
            elapsed = time.time() - stats["start_time"]
            avg_time = elapsed / stats["processed"]
            remaining = (stats["total_chunks"] - stats["processed"]) * avg_time
            
            print(f"\nCheckpoint saved: {stats['processed']}/{stats['total_chunks']} chunks")
            print(f"Triples extracted: {stats['total_triples']}")
            print(f"ETA: {remaining/60:.1f} minutes")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_CALLS)
    
    # Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    elapsed_time = time.time() - stats["start_time"]
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Total chunks processed: {stats['processed']}")
    print(f"Failed extractions: {stats['failed']}")
    print(f"Total triples extracted: {stats['total_triples']}")
    print(f"Average triples per chunk: {stats['total_triples'] / max(1, stats['processed']):.1f}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per chunk: {elapsed_time/stats['processed']:.2f}s")
    print(f"\nOutput saved to: {OUTPUT_JSON}")
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_JSON):
        os.remove(CHECKPOINT_JSON)
        print("Checkpoint file removed")
    
    # Generate summary report
    generate_summary_report(all_results)

def generate_summary_report(triples: List[Dict]):
    """Generate summary statistics for hackathon demo"""
    from collections import Counter
    
    if not triples:
        print("\nNo triples to analyze")
        return
    
    subjects = [t["subject"] for t in triples]
    predicates = [t["predicate"] for t in triples]
    objects = [t["object"] for t in triples]
    
    report = {
        "total_triples": len(triples),
        "unique_subjects": len(set(subjects)),
        "unique_predicates": len(set(predicates)),
        "unique_objects": len(set(objects)),
        "unique_papers": len(set(t["title"] for t in triples)),
        "top_subjects": dict(Counter(subjects).most_common(10)),
        "top_predicates": dict(Counter(predicates).most_common(10)),
        "top_objects": dict(Counter(objects).most_common(10))
    }
    
    # Save report
    with open("kg_summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 50)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("=" * 50)
    print(f"Total Relationships: {report['total_triples']}")
    print(f"Unique Entities: {report['unique_subjects'] + report['unique_objects']}")
    print(f"Relationship Types: {report['unique_predicates']}")
    print(f"Papers Covered: {report['unique_papers']}")
    
    print("\nTop 5 Most Common Relationships:")
    for pred, count in list(report['top_predicates'].items())[:5]:
        print(f"  {pred}: {count} occurrences")
    
    print("\nTop 5 Most Common Entities:")
    subject_counter = Counter(subjects)
    object_counter = Counter(objects)
    all_entities = subject_counter + object_counter
    for entity, count in all_entities.most_common(5):
        print(f"  {entity}: {count} occurrences")
    
    print(f"\nSummary report saved to: kg_summary_report.json")

if __name__ == "__main__":
    main()