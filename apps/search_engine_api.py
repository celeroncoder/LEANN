"""
FastAPI-based semantic search engine for LEANN
Processes files from data/ folder and provides semantic search results
Includes analytics, auto-sync, and comprehensive logging
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import our semantic search engine
from semantic_search import SemanticSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_semantic: bool = True
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source_file: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    timestamp: str

class AnalyticsData(BaseModel):
    total_queries: int
    avg_response_time: float
    popular_queries: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    index_stats: Dict[str, Any]

class IndexStatus(BaseModel):
    total_files: int
    indexed_files: int
    last_update: str
    index_size_mb: float
    status: str

# Global variables for search engine state
search_index = {}
file_contents = {}
semantic_engine = SemanticSearchEngine()  # Initialize semantic search engine
analytics = {
    "queries": [],
    "total_queries": 0,
    "response_times": [],
}
data_folder = Path("/home/runner/work/LEANN/LEANN/data")
index_status = {
    "total_files": 0,
    "indexed_files": 0,
    "last_update": datetime.now().isoformat(),
    "index_size_mb": 0.0,
    "status": "initializing"
}

# Initialize FastAPI app
app = FastAPI(
    title="LEANN Semantic Search Engine",
    description="FastAPI-based search engine with real-time indexing and analytics",
    version="1.0.0"
)

class FileWatcher(FileSystemEventHandler):
    """File system event handler for monitoring data folder changes"""
    
    def __init__(self, indexer_func):
        self.indexer_func = indexer_func
        super().__init__()
    
    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            # Schedule indexing in a thread-safe way
            asyncio.run_coroutine_threadsafe(
                self.indexer_func(Path(event.src_path)), 
                asyncio.get_event_loop() if hasattr(asyncio, '_get_running_loop') else None
            )
    
    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"File created: {event.src_path}")
            # Schedule indexing in a thread-safe way
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.indexer_func(Path(event.src_path)), loop)
                else:
                    # If no loop is running, we'll catch this in the reindex endpoint
                    logger.warning("No event loop running, file will be indexed on next reindex")
            except RuntimeError:
                logger.warning("No event loop available, file will be indexed on next reindex")
    
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            # Remove from index
            file_path = str(event.src_path)
            if file_path in file_contents:
                del file_contents[file_path]
                semantic_engine.remove_document(file_path)
                update_index_status()

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics for analytics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }

def calculate_index_size() -> float:
    """Calculate approximate index size in MB"""
    total_size = 0
    for content in file_contents.values():
        total_size += len(str(content).encode('utf-8'))
    return total_size / (1024 * 1024)  # Convert to MB

def update_index_status():
    """Update global index status"""
    global index_status
    index_status.update({
        "total_files": len(list(data_folder.rglob("*.*"))),
        "indexed_files": len(file_contents),
        "last_update": datetime.now().isoformat(),
        "index_size_mb": calculate_index_size(),
        "status": "ready" if file_contents else "empty"
    })

async def process_file(file_path: Path) -> Optional[str]:
    """Process a single file and extract text content"""
    try:
        if not file_path.exists():
            return None
            
        # Handle different file types
        content = ""
        if file_path.suffix.lower() == '.txt':
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        elif file_path.suffix.lower() == '.md':
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        elif file_path.suffix.lower() == '.pdf':
            # For now, just note it's a PDF - would need PDF parsing library
            content = f"PDF file: {file_path.name}"
        else:
            # Try to read as text
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except:
                content = f"Binary file: {file_path.name}"
        
        return content
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

async def index_file(file_path: Path):
    """Index a single file"""
    try:
        content = await process_file(file_path)
        if content:
            file_data = {
                "content": content,
                "metadata": {
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                },
                "indexed_at": datetime.now().isoformat()
            }
            file_contents[str(file_path)] = file_data
            
            # Add to semantic search index
            semantic_engine.add_document(str(file_path), content, file_data["metadata"])
            
            update_index_status()
            logger.info(f"Indexed file: {file_path}")
    except Exception as e:
        logger.error(f"Error indexing file {file_path}: {e}")

async def reindex_all_files():
    """Reindex all files in the data folder"""
    logger.info("Starting full reindexing...")
    global index_status
    index_status["status"] = "indexing"
    
    if not data_folder.exists():
        logger.warning(f"Data folder {data_folder} does not exist")
        index_status["status"] = "error"
        return
    
    # Clear existing indexes
    semantic_engine.clear_index()
    
    files = list(data_folder.rglob("*.*"))
    logger.info(f"Found {len(files)} files to index")
    
    for file_path in files:
        if file_path.is_file():
            await index_file(file_path)
    
    index_status["status"] = "ready"
    logger.info(f"Reindexing complete. Indexed {len(file_contents)} files")

def hybrid_search(query: str, top_k: int = 5, use_semantic: bool = True) -> List[SearchResult]:
    """Hybrid search combining semantic and keyword-based search"""
    results = []
    
    # Perform semantic search if enabled and available
    if use_semantic:
        semantic_results = semantic_engine.search(query, top_k)
        for doc_id, score, text, metadata in semantic_results:
            results.append(SearchResult(
                id=doc_id,
                text=text[:500] + "..." if len(text) > 500 else text,
                score=score,
                metadata=metadata,
                source_file=doc_id
            ))
    
    # If no semantic results or semantic disabled, fall back to keyword search
    if not results:
        results = simple_text_search(query, top_k)
    
    return results

def simple_text_search(query: str, top_k: int = 5) -> List[SearchResult]:
    """Simple text-based search (fallback when no embeddings available)"""
    query_lower = query.lower()
    results = []
    
    for file_path, data in file_contents.items():
        content = data["content"].lower()
        # Simple scoring based on term frequency
        score = content.count(query_lower) / len(content.split()) if content else 0.0
        
        if score > 0:
            results.append(SearchResult(
                id=file_path,
                text=data["content"][:500] + "..." if len(data["content"]) > 500 else data["content"],
                score=score,
                metadata=data["metadata"],
                source_file=file_path
            ))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    logger.info("Starting LEANN Search Engine...")
    
    # Create data folder if it doesn't exist
    data_folder.mkdir(exist_ok=True)
    
    # Initial indexing
    await reindex_all_files()
    
    # Setup file watcher
    event_handler = FileWatcher(index_file)
    observer = Observer()
    observer.schedule(event_handler, str(data_folder), recursive=True)
    observer.start()
    
    # Store observer in app state for cleanup
    app.state.observer = observer
    
    logger.info("Search engine initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app.state, 'observer'):
        app.state.observer.stop()
        app.state.observer.join()
    logger.info("Search engine shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LEANN Semantic Search Engine",
        "version": "1.0.0",
        "status": index_status["status"],
        "endpoints": [
            "/search",
            "/analytics",
            "/status",
            "/reindex",
            "/health"
        ]
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Perform semantic search on indexed documents"""
    start_time = time.time()
    
    try:
        # Log the search query
        logger.info(f"Search query: {request.query}")
        
        # Perform search (using hybrid search with semantic capabilities)
        results = hybrid_search(request.query, request.top_k, request.use_semantic)
        
        processing_time = time.time() - start_time
        
        # Update analytics
        analytics["queries"].append({
            "query": request.query,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "results_count": len(results)
        })
        analytics["total_queries"] += 1
        analytics["response_times"].append(processing_time)
        
        # Keep only last 1000 queries for memory management
        if len(analytics["queries"]) > 1000:
            analytics["queries"] = analytics["queries"][-1000:]
        if len(analytics["response_times"]) > 1000:
            analytics["response_times"] = analytics["response_times"][-1000:]
        
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Search completed in {processing_time:.3f}s, found {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/analytics", response_model=AnalyticsData)
async def get_analytics():
    """Get search analytics and system metrics"""
    # Calculate popular queries
    query_counts = {}
    for query_data in analytics["queries"]:
        query = query_data["query"]
        query_counts[query] = query_counts.get(query, 0) + 1
    
    popular_queries = [
        {"query": query, "count": count} 
        for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    # Calculate average response time
    avg_response_time = (
        sum(analytics["response_times"]) / len(analytics["response_times"])
        if analytics["response_times"] else 0.0
    )
    
    return AnalyticsData(
        total_queries=analytics["total_queries"],
        avg_response_time=avg_response_time,
        popular_queries=popular_queries,
        system_metrics=get_system_metrics(),
        index_stats={**index_status, "semantic_stats": semantic_engine.get_stats()}
    )

@app.get("/status", response_model=IndexStatus)
async def get_status():
    """Get current index status"""
    update_index_status()
    return IndexStatus(**index_status)

@app.post("/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    """Trigger a full reindex of all files"""
    background_tasks.add_task(reindex_all_files)
    return {"message": "Reindexing started", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": get_system_metrics(),
        "index": index_status
    }

@app.get("/files")
async def list_files():
    """List all indexed files"""
    files_info = []
    for file_path, data in file_contents.items():
        files_info.append({
            "path": file_path,
            "metadata": data["metadata"],
            "indexed_at": data["indexed_at"],
            "content_preview": data["content"][:200] + "..." if len(data["content"]) > 200 else data["content"]
        })
    
    return {
        "total_files": len(files_info),
        "files": files_info
    }

if __name__ == "__main__":
    uvicorn.run(
        "search_engine_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )