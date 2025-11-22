"""
Vector Database Manager - PDF-focused interface

This project now limits vector database usage to PDF Analyzer only. To avoid
accidentally adding resumes, job descriptions, or interview question banks,
most non-PDF helper methods have been restricted. The manager still provides
methods for adding/searching 'knowledge' items (used for PDF chunks/resources),
listing documents, and inspecting metadata.

Use `purge_non_pdf_content(dry_run=True)` to preview non-PDF rows that can be
removed. To actually delete them, call with `dry_run=False`.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import vector database at module level to avoid circular imports
try:
    from utils.vector_database import VectorDatabase
except ImportError:
    VectorDatabase = None

class VectorDBManager:
    """Unified manager for vector database operations across all features"""
    
    def __init__(self):
        self.vector_db = None
        self.initialized = False
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        try:
            if VectorDatabase:
                self.vector_db = VectorDatabase("placement_bot_unified")
                self.initialized = True
                logger.info("✅ Vector Database Manager initialized successfully")
            else:
                logger.error("❌ VectorDatabase class not available")
                self.initialized = False
        except ImportError as e:
            logger.error(f"❌ Failed to import VectorDatabase: {e}")
            self.initialized = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize VectorDatabase: {e}")
            self.initialized = False
    
    def is_available(self) -> bool:
        return bool(self.initialized and self.vector_db is not None)

    def seed_templates(self, templates: Optional[List[Dict[str, Any]]] = None) -> int:
        """Seed predefined templates into the knowledge index.

        Each template will be stored as a knowledge item with metadata:
        - type: 'template'
        - template_for: feature key (e.g., 'resume_analysis', 'pdf_summary')
        - title: human-readable title
        - version: optional version string
        - template: the full template text (duplicated in metadata for retrieval)

        Returns the number of templates successfully upserted (added or already present).
        """
        if not self.is_available():
            logger.warning("Vector database not available; skipping template seeding")
            return 0
        try:
            if templates is None:
                try:
                    from utils import templates as tpl
                    templates = tpl.TEMPLATES  # type: ignore[attr-defined]
                except Exception:
                    templates = []
            count = 0
            for t in templates or []:
                text = (t.get('template') or t.get('text') or '').strip()
                title = t.get('title') or f"Template: {t.get('template_for','unknown')}"
                if not text:
                    continue
                md = {
                    'type': 'template',
                    'template_for': t.get('template_for', 'general'),
                    'title': title,
                    'version': t.get('version', 'v1'),
                    'excerpt': text[:600],
                    'template': text,
                }
                try:
                    self.add_knowledge_item(f"{title}\n\n{text}", md)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to seed template '{title}': {e}")
            if count:
                logger.info(f"Seeded {count} templates into vector DB")
            return count
        except Exception as e:
            logger.error(f"Error seeding templates: {e}")
            return 0

    def search_templates(self, template_for: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search templates for a given feature key using the knowledge index and filter by metadata."""
        try:
            results = self.search_knowledge(template_for, top_k=top_k * 3)
            return [r for r in results if r.get('metadata', {}).get('type') == 'template' and r.get('metadata', {}).get('template_for') == template_for]
        except Exception:
            return []

    # Interview Question Seeding
    def seed_interview_questions(self, question_bank: Optional[List[Dict[str, Any]]] = None, overwrite: bool = False) -> int:
        """Seed curated technical and HR interview Q&A into the knowledge index.

        Stores items as knowledge with metadata:
        - type: 'technical_question' or 'hr_question'
        - title: the question text
        - category, tags, difficulty, skill
        - excerpt: short snippet for quick context
        Returns count inserted.
        """
        # Interview question seeding is disabled under PDF-only policy
        logger.warning("Interview question seeding is disabled. Vector DB is PDF-only.")
        return 0
        try:
            if question_bank is None:
                try:
                    from utils.question_bank import INTERVIEW_QUESTIONS  # base bank
                except Exception:
                    INTERVIEW_QUESTIONS = []  # type: ignore
                try:
                    from utils.question_bank_bulk import BULK_INTERVIEW_QUESTIONS  # optional bulk
                except Exception:
                    BULK_INTERVIEW_QUESTIONS = []  # type: ignore
                question_bank = list(INTERVIEW_QUESTIONS) + list(BULK_INTERVIEW_QUESTIONS)
            count = 0
            for item in question_bank or []:
                q = (item.get('question') or '').strip()
                a = (item.get('answer') or '').strip()
                if not q or not a:
                    continue
                category = (item.get('category') or 'technical').lower()
                md_type = 'technical_question' if category == 'technical' else 'hr_question'
                text = f"Q: {q}\nA: {a}"
                excerpt = (a[:600] if a else text[:600])
                metadata = {
                    'type': md_type,
                    'title': q,
                    'category': category,
                    'tags': item.get('tags', []),
                    'difficulty': item.get('difficulty', 'medium'),
                    'skill': item.get('skill', ''),
                    'excerpt': excerpt,
                    'question': q,
                }
                try:
                    # No explicit de-dup; rely on vector similarity and underlying DB if supported
                    self.add_knowledge_item(text, metadata)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to seed interview question '{q[:40]}...': {e}")
            if count:
                logger.info(f"Seeded {count} interview questions into vector DB")
            return count
        except Exception as e:
            logger.error(f"Error seeding interview questions: {e}")
            return 0
    
    # Knowledge Base Operations
    def add_knowledge_item(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Add a knowledge item to the database"""
        if not self.is_available():
            logger.warning("Vector database not available")
            return None
            
        try:
            doc_id = self.vector_db.add_knowledge_item(text, metadata)
            logger.info(f"Added knowledge item with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
            return None
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for knowledge items"""
        if not self.is_available():
            logger.warning("Vector database not available")
            return []
            
        try:
            results = self.vector_db.search_knowledge(query, top_k)
            logger.info(f"Found {len(results)} knowledge items for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    # Resume Operations
    def add_resume(self, resume_text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Add a resume to the database"""
        # Disabled: vector DB is restricted to PDF content only.
        logger.warning("add_resume is disabled: vector DB restricted to PDF content only")
        return None
    
    def search_similar_resumes(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar resumes"""
        logger.warning("search_similar_resumes is disabled: vector DB restricted to PDF content only")
        return []
    
    # Job Description Operations
    def add_job_description(self, job_text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Add a job description to the database"""
        logger.warning("add_job_description is disabled: vector DB restricted to PDF content only")
        return None
    
    def search_matching_jobs(self, resume_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for matching jobs"""
        logger.warning("search_matching_jobs is disabled: vector DB restricted to PDF content only")
        return []
    
    # Technical Interview Operations
    def add_technical_question(self, question_text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Add a technical interview question"""
        logger.warning("add_technical_question is disabled: vector DB restricted to PDF content only")
        return None
    
    def search_technical_questions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for technical interview questions"""
        logger.warning("search_technical_questions is disabled: vector DB restricted to PDF content only")
        return []
    
    # HR Interview Operations
    def add_hr_question(self, question_text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Add an HR interview question"""
        logger.warning("add_hr_question is disabled: vector DB restricted to PDF content only")
        return None
    
    def search_hr_questions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for HR interview questions"""
        logger.warning("search_hr_questions is disabled: vector DB restricted to PDF content only")
        return []
    
    # Statistics and Utilities
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_available():
            logger.warning("Vector database not available")
            return {}
        try:
            stats = self.vector_db.get_database_stats()
            # Only expose stats relevant to PDF knowledge
            return {
                'knowledge_vectors': stats.get('knowledge_vectors', 0),
                'resumes_vectors': 0,
                'jobs_vectors': 0,
                'documents_by_type': {k: v for k, v in stats.items() if k.endswith('_count')}
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if not self.is_available():
            logger.warning("Vector database not available")
            return None
            
        try:
            doc = self.vector_db.get_document_by_id(doc_id)
            logger.info(f"Retrieved document with ID: {doc_id}")
            return doc
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None

    def purge_non_pdf_content(self, dry_run: bool = True) -> List[Dict[str, Any]]:
        """List or remove non-PDF content from the metadata DB.

        By default (`dry_run=True`) returns a list of candidate rows that would be
        removed. If `dry_run=False` the rows will be deleted from the `documents`
        table. Note: FAISS vector cleanup is not performed automatically.
        """
        if not self.is_available():
            logger.warning("Vector database not available")
            return []
        try:
            cursor = self.conn.cursor()
            # Select rows whose metadata does not contain a document_id (likely non-PDF knowledge)
            rows = cursor.execute(
                "SELECT id, doc_id, doc_type, title, metadata, created_at FROM documents WHERE (doc_type != 'knowledge' OR metadata NOT LIKE '%\"document_id\"%') ORDER BY created_at DESC"
            ).fetchall()

            candidates = []
            for row in rows:
                row_id, doc_id, doc_type, title, metadata_json, created_at = row
                try:
                    md = json.loads(metadata_json) if metadata_json else {}
                except Exception:
                    md = {}
                candidates.append({
                    'row_id': row_id,
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'title': title,
                    'metadata': md,
                    'created_at': created_at
                })

            if dry_run:
                logger.info(f"Purge dry-run: {len(candidates)} non-PDF rows found")
                return candidates

            # Perform deletion
            deleted = []
            for c in candidates:
                try:
                    cursor.execute("DELETE FROM documents WHERE id = ?", (c['row_id'],))
                    deleted.append(c)
                except Exception as e:
                    logger.error(f"Failed to delete row {c['row_id']}: {e}")
            self.conn.commit()
            logger.info(f"Purged {len(deleted)} non-PDF rows from metadata DB")
            return deleted
        except Exception as e:
            logger.error(f"Error during purge_non_pdf_content: {e}")
            return []

    def list_documents(self, filter_has_document_id: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """List documents stored in the metadata DB.

        By default, returns knowledge items that contain a 'document_id' field in their metadata
        (which corresponds to stored PDF chunks / documents).
        """
        if not self.is_available():
            logger.warning("Vector database not available")
            return []
        try:
            cursor = self.conn.cursor()
            if filter_has_document_id:
                # Simple JSON text search on metadata column for document_id key
                rows = cursor.execute(
                    "SELECT doc_id, doc_type, title, metadata, created_at FROM documents WHERE doc_type = 'knowledge' AND metadata LIKE '%\"document_id\"%' ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = cursor.execute(
                    "SELECT doc_id, doc_type, title, metadata, created_at FROM documents ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            results = []
            for doc_id, doc_type, title, metadata_json, created_at in rows:
                try:
                    md = json.loads(metadata_json) if metadata_json else {}
                except Exception:
                    md = {}
                results.append({
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'title': title,
                    'metadata': md,
                    'created_at': created_at
                })
            return results
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

# Global instance
vector_db_manager = VectorDBManager()

# Convenience functions for direct import
def is_vector_db_available():
    """Check if vector database is available"""
    return vector_db_manager.is_available()

def add_knowledge_item(text: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Add knowledge item"""
    return vector_db_manager.add_knowledge_item(text, metadata)

def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search knowledge"""
    return vector_db_manager.search_knowledge(query, top_k)

def add_resume(resume_text: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Add resume"""
    return vector_db_manager.add_resume(resume_text, metadata)

def search_similar_resumes(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search similar resumes"""
    return vector_db_manager.search_similar_resumes(query_text, top_k)

def add_job_description(job_text: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Add job description"""
    return vector_db_manager.add_job_description(job_text, metadata)

def search_matching_jobs(resume_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search matching jobs"""
    return vector_db_manager.search_matching_jobs(resume_text, top_k)

def add_technical_question(question_text: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Add technical question"""
    return vector_db_manager.add_technical_question(question_text, metadata)

def search_technical_questions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search technical questions"""
    return vector_db_manager.search_technical_questions(query, top_k)

def get_database_stats() -> Dict[str, Any]:
    """Get database stats"""
    return vector_db_manager.get_database_stats()

def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Get document by ID"""
    return vector_db_manager.get_document_by_id(doc_id)

def list_documents(filter_has_document_id: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience wrapper to list documents from the vector DB manager."""
    return vector_db_manager.list_documents(filter_has_document_id=filter_has_document_id, limit=limit)

def ensure_templates_seeded() -> int:
    """Ensure core templates are present in the vector DB."""
    return vector_db_manager.seed_templates()

def search_templates(template_for: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search templates by feature key."""
    return vector_db_manager.search_templates(template_for, top_k)

def ensure_interview_questions_seeded() -> int:
    """Ensure curated interview questions are present in the vector DB."""
    return vector_db_manager.seed_interview_questions()

# Export the VectorDBManager class and utility functions
__all__ = [
    'VectorDBManager',
    'vector_db_manager',
    'is_vector_db_available',
    'add_knowledge_item',
    'search_knowledge',
    'add_resume',
    'search_similar_resumes',
    'add_job_description',
    'search_matching_jobs',
    'add_technical_question',
    'search_technical_questions',
    'get_database_stats',
    'get_document_by_id',
    'list_documents',
    'ensure_templates_seeded',
    'search_templates',
    'ensure_interview_questions_seeded',
]