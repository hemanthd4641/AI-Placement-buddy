"""
Enhanced Vector Database for AI Placement Mentor Bot
Handles resume embeddings, job descriptions, and knowledge base with FAISS indexing
"""

import os
import json
import sqlite3
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import faiss
from utils.embeddings import embed_texts

class VectorDatabase:
    """Enhanced vector database for storing and searching document embeddings"""
    
    def __init__(self, db_name: str = "placement_vectors"):
        self.db_name = db_name
        self.db_dir = Path("vector_db")
        self.db_dir.mkdir(exist_ok=True)
        
        # Use embeddings adapter (ModelManager / Gemini preferred)
        # Default embedding dimension for cloud embeddings (Gemini) is 768
        self.dimension = 768
        
        # Initialize separate indexes for different document types
        self.resumes_index = None
        self.jobs_index = None
        self.knowledge_index = None
        
        # FAISS index file paths
        self.resumes_index_path = self.db_dir / f"{db_name}_resumes.index"
        self.jobs_index_path = self.db_dir / f"{db_name}_jobs.index"
        self.knowledge_index_path = self.db_dir / f"{db_name}_knowledge.index"
        
        # SQLite database for metadata
        self.db_path = self.db_dir / f"{db_name}_metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        # Initialize database
        self._initialize_database()
        self._load_or_create_indexes()
    
    def _initialize_database(self):
        """Initialize SQLite database schema"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                doc_type TEXT NOT NULL,
                title TEXT,
                content_hash TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)
        ''')
        
        self.conn.commit()
    
    def _load_or_create_indexes(self):
        """Load existing FAISS indexes or create new ones"""
        # Load or create resumes index
        if self.resumes_index_path.exists():
            try:
                self.resumes_index = faiss.read_index(str(self.resumes_index_path))
                print(f"✅ Loaded resumes index with {self.resumes_index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading resumes index: {e}")
                self.resumes_index = faiss.IndexFlatIP(self.dimension)
                print("🆕 Created new resumes index")
        else:
            self.resumes_index = faiss.IndexFlatIP(self.dimension)
            print("🆕 Created new resumes index")
        
        # Load or create jobs index
        if self.jobs_index_path.exists():
            try:
                self.jobs_index = faiss.read_index(str(self.jobs_index_path))
                print(f"✅ Loaded jobs index with {self.jobs_index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading jobs index: {e}")
                self.jobs_index = faiss.IndexFlatIP(self.dimension)
                print("🆕 Created new jobs index")
        else:
            self.jobs_index = faiss.IndexFlatIP(self.dimension)
            print("🆕 Created new jobs index")
        
        # Load or create knowledge index
        if self.knowledge_index_path.exists():
            try:
                self.knowledge_index = faiss.read_index(str(self.knowledge_index_path))
                print(f"✅ Loaded knowledge index with {self.knowledge_index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading knowledge index: {e}")
                self.knowledge_index = faiss.IndexFlatIP(self.dimension)
                print("🆕 Created new knowledge index")
        else:
            self.knowledge_index = faiss.IndexFlatIP(self.dimension)
            print("🆕 Created new knowledge index")
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID from content"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with fallback to zero vector"""
        try:
            emb = embed_texts([text], dim=self.dimension)
            # embed_texts may return single row or array
            emb = np.array(emb, dtype='float32')
            if emb.ndim == 2:
                emb = emb[0]
            # Ensure shape
            if emb.size != self.dimension:
                # Resize/pad or truncate
                arr = np.zeros(self.dimension, dtype='float32')
                arr[:min(self.dimension, emb.size)] = emb.flatten()[:self.dimension]
                emb = arr
            # Normalize
            try:
                faiss.normalize_L2(emb.reshape(1, -1))
            except Exception:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb.astype('float32').reshape(1, -1)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros((1, self.dimension), dtype='float32')
    
    def add_resume(self, resume_text: str, metadata: Dict[str, Any]) -> str:
        """Add resume to vector database"""
        doc_id = self._generate_doc_id(resume_text)
        
        # Check if already exists
        existing = self.conn.execute(
            "SELECT id FROM documents WHERE doc_id = ? AND doc_type = 'resume'",
            (doc_id,)
        ).fetchone()
        
        if existing:
            print(f"Resume already exists with ID: {doc_id}")
            return doc_id
        
        # Generate embedding
        embedding = self._get_embedding(resume_text)
        
        # Add to FAISS index
        self.resumes_index.add(embedding)
        
        # Store metadata in SQLite
        metadata_json = json.dumps(metadata)
        content_hash = hashlib.sha256(resume_text.encode()).hexdigest()
        
        self.conn.execute(
            "INSERT INTO documents (doc_id, doc_type, title, content_hash, metadata) VALUES (?, ?, ?, ?, ?)",
            (doc_id, 'resume', metadata.get('title', 'Resume'), content_hash, metadata_json)
        )
        self.conn.commit()
        
        # Save index
        faiss.write_index(self.resumes_index, str(self.resumes_index_path))
        
        print(f"✅ Added resume to database: {doc_id}")
        return doc_id
    
    def add_job_description(self, job_text: str, metadata: Dict[str, Any]) -> str:
        """Add job description to vector database"""
        doc_id = self._generate_doc_id(job_text)
        
        # Check if already exists
        existing = self.conn.execute(
            "SELECT id FROM documents WHERE doc_id = ? AND doc_type = 'job'",
            (doc_id,)
        ).fetchone()
        
        if existing:
            print(f"Job description already exists with ID: {doc_id}")
            return doc_id
        
        # Generate embedding
        embedding = self._get_embedding(job_text)
        
        # Add to FAISS index
        self.jobs_index.add(embedding)
        
        # Store metadata in SQLite
        metadata_json = json.dumps(metadata)
        content_hash = hashlib.sha256(job_text.encode()).hexdigest()
        
        self.conn.execute(
            "INSERT INTO documents (doc_id, doc_type, title, content_hash, metadata) VALUES (?, ?, ?, ?, ?)",
            (doc_id, 'job', metadata.get('title', 'Job Description'), content_hash, metadata_json)
        )
        self.conn.commit()
        
        # Save index
        faiss.write_index(self.jobs_index, str(self.jobs_index_path))
        
        print(f"✅ Added job description to database: {doc_id}")
        return doc_id
    
    def add_knowledge_item(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add knowledge item to vector database"""
        doc_id = self._generate_doc_id(text)
        
        # Check if already exists
        existing = self.conn.execute(
            "SELECT id FROM documents WHERE doc_id = ? AND doc_type = 'knowledge'",
            (doc_id,)
        ).fetchone()
        
        if existing:
            print(f"Knowledge item already exists with ID: {doc_id}")
            return doc_id
        
        # Generate embedding
        embedding = self._get_embedding(text)
        
        # Add to FAISS index
        self.knowledge_index.add(embedding)
        
        # Store metadata in SQLite
        metadata_json = json.dumps(metadata)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        self.conn.execute(
            "INSERT INTO documents (doc_id, doc_type, title, content_hash, metadata) VALUES (?, ?, ?, ?, ?)",
            (doc_id, 'knowledge', metadata.get('title', 'Knowledge Item'), content_hash, metadata_json)
        )
        self.conn.commit()
        
        # Save index
        faiss.write_index(self.knowledge_index, str(self.knowledge_index_path))
        
        print(f"✅ Added knowledge item to database: {doc_id}")
        return doc_id
    
    def search_resumes(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar resumes"""
        if self.resumes_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Search in FAISS index
        distances, indices = self.resumes_index.search(query_embedding, min(top_k, self.resumes_index.ntotal))
        
        # Retrieve metadata from SQLite
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index != -1 and distance > 0:  # Valid result
                # Get document metadata
                cursor = self.conn.execute(
                    "SELECT doc_id, metadata FROM documents WHERE doc_type = 'resume' LIMIT 1 OFFSET ?",
                    (int(index),)
                )
                row = cursor.fetchone()
                if row:
                    doc_id, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    results.append({
                        'doc_id': doc_id,
                        'score': float(distance),
                        'metadata': metadata
                    })
        
        return results
    
    def search_jobs(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar job descriptions"""
        if self.jobs_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Search in FAISS index
        distances, indices = self.jobs_index.search(query_embedding, min(top_k, self.jobs_index.ntotal))
        
        # Retrieve metadata from SQLite
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index != -1 and distance > 0:  # Valid result
                # Get document metadata
                cursor = self.conn.execute(
                    "SELECT doc_id, metadata FROM documents WHERE doc_type = 'job' LIMIT 1 OFFSET ?",
                    (int(index),)
                )
                row = cursor.fetchone()
                if row:
                    doc_id, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    results.append({
                        'doc_id': doc_id,
                        'score': float(distance),
                        'metadata': metadata
                    })
        
        return results
    
    def search_knowledge(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar knowledge items"""
        if self.knowledge_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Search in FAISS index
        distances, indices = self.knowledge_index.search(query_embedding, min(top_k, self.knowledge_index.ntotal))
        
        # Retrieve metadata from SQLite
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index != -1 and distance > 0:  # Valid result
                # Get document metadata
                cursor = self.conn.execute(
                    "SELECT doc_id, metadata FROM documents WHERE doc_type = 'knowledge' LIMIT 1 OFFSET ?",
                    (int(index),)
                )
                row = cursor.fetchone()
                if row:
                    doc_id, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    results.append({
                        'doc_id': doc_id,
                        'score': float(distance),
                        'metadata': metadata
                    })
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        result = self.conn.execute(
            "SELECT doc_type, title, metadata, created_at FROM documents WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        
        if result:
            doc_type, title, metadata_json, created_at = result
            return {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'title': title,
                'metadata': json.loads(metadata_json),
                'created_at': created_at
            }
        return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # Count documents by type
        result = self.conn.execute(
            "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"
        ).fetchall()
        
        for doc_type, count in result:
            stats[f"{doc_type}_count"] = count
        
        # Vector index sizes
        stats['resumes_vectors'] = self.resumes_index.ntotal if self.resumes_index else 0
        stats['jobs_vectors'] = self.jobs_index.ntotal if self.jobs_index else 0
        stats['knowledge_vectors'] = self.knowledge_index.ntotal if self.knowledge_index else 0
        
        return stats
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()