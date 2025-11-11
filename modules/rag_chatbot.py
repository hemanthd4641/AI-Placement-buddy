"""RAG Chatbot

LLM + Vector DB augmented QA for technical skills and HR interview topics.
Resilient to missing optional dependencies; degrades gracefully.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Lazy optional imports similar to other modules
try:
    from utils.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except Exception:
    ModelManager = None  # type: ignore
    MODEL_MANAGER_AVAILABLE = False

try:
    from utils.vector_db_manager import VectorDBManager, is_vector_db_available
    VECTOR_DB_AVAILABLE = True
except Exception:
    VectorDBManager = None  # type: ignore
    is_vector_db_available = lambda: False  # type: ignore
    VECTOR_DB_AVAILABLE = False

class RAGChatbot:
    """Retrieval-Augmented chatbot for placement prep (tech + HR)."""

    def __init__(self) -> None:
        # Initialize model manager
        if MODEL_MANAGER_AVAILABLE and ModelManager is not None:
            try:
                self.model_manager = ModelManager()
            except Exception:
                self.model_manager = None
        else:
            self.model_manager = None

        # Initialize vector DB manager
        if VECTOR_DB_AVAILABLE and is_vector_db_available():
            try:
                self.vector_db_manager = VectorDBManager()
            except Exception:
                self.vector_db_manager = None
        else:
            self.vector_db_manager = None

    def _search_sources(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search across knowledge base for mixed content (tech Qs, HR, PDFs, resources)."""
        results: List[Dict[str, Any]] = []
        if not self.vector_db_manager:
            return results
        try:
            # General knowledge
            kb = self.vector_db_manager.search_knowledge(query, top_k)
            results.extend(kb)
        except Exception:
            pass

        # Try specific typed helpers if available on manager
        try:
            if hasattr(self.vector_db_manager, 'search_technical_questions'):
                tqs = self.vector_db_manager.search_technical_questions(query, top_k=top_k)
                results.extend(tqs)
        except Exception:
            pass
        try:
            if hasattr(self.vector_db_manager, 'search_hr_questions'):
                hrs = self.vector_db_manager.search_hr_questions(query, top_k=top_k)
                results.extend(hrs)
        except Exception:
            pass

        # Deduplicate by doc_id keeping highest score
        dedup: Dict[str, Dict[str, Any]] = {}
        for r in results:
            doc_id = r.get('doc_id') or ''
            if not doc_id:
                continue
            if doc_id not in dedup or r.get('score', 0) > dedup[doc_id].get('score', 0):
                dedup[doc_id] = r
        return list(dedup.values())

    def _compose_context(self, question: str, sources: List[Dict[str, Any]], max_chars: int = 1500) -> Tuple[str, List[Dict[str, Any]]]:
        """Build a compact context string and a cleaned list of citations."""
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        for idx, r in enumerate(sorted(sources, key=lambda x: x.get('score', 0), reverse=True)):
            md = r.get('metadata', {}) or {}
            title = md.get('title') or md.get('file_name') or md.get('skill') or md.get('type') or f"Doc {idx+1}"
            excerpt = (md.get('excerpt') or '')
            # If we lack an excerpt, try to synthesize from available metadata
            if not excerpt:
                # Join a few known fields to provide some context
                for k in ['description', 'summary', 'content', 'question']:
                    if md.get(k):
                        excerpt = str(md[k])
                        break
            snippet = (excerpt or '')[:500]
            if snippet:
                parts.append(f"- {title}: {snippet}")
                citations.append({
                    'title': title,
                    'type': md.get('type', 'knowledge'),
                    'score': r.get('score', 0),
                })
            if len("\n".join(parts)) > max_chars:
                break
        context = "\n".join(parts)[:max_chars]
        # Always include the question at the end to focus the model
        context_block = f"[Question]\n{question}\n\n[Retrieved Context]\n{context}" if context else f"[Question]\n{question}"
        return context_block, citations

    def _build_prompt(self, question: str, chat_history: List[Dict[str, str]], context_block: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        # Keep last 6 turns compact
        history_lines: List[str] = []
        for turn in chat_history[-6:]:
            role = turn.get('role', 'user')
            content = (turn.get('content') or '').strip().replace('\n', ' ')
            if not content:
                continue
            if role == 'user':
                history_lines.append(f"User: {content}")
            else:
                history_lines.append(f"Assistant: {content}")
        history_text = "\n".join(history_lines)
        profile = ''
        if user_profile:
            try:
                role = user_profile.get('target_role') or user_profile.get('desired_role') or ''
                exp = user_profile.get('experience_level') or user_profile.get('years_experience') or ''
                skills = ', '.join(user_profile.get('skills', [])[:10]) if isinstance(user_profile.get('skills'), list) else ''
                profile = f"Target Role: {role}\nExperience: {exp}\nSkills: {skills}".strip()
            except Exception:
                profile = ''
        system_instructions = (
            "You are an expert placement mentor chatbot. Answer questions about technical skills, data structures/algorithms, software engineering, and HR interviews. "
            "Use only the retrieved context when citing facts. If the answer is not in the context, say you cannot find it and propose how to search or clarify. Be concise and practical."
        )
        prompt = (
            f"{system_instructions}\n\n"
            f"[User Profile]\n{profile}\n\n" if profile else f"{system_instructions}\n\n"
        )
        if history_text:
            prompt += f"[Recent Conversation]\n{history_text}\n\n"
        prompt += f"{context_block}\n\nAnswer:"  # Direct the model to answer now
        return prompt

    def answer(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chat_history = chat_history or []
        # Retrieve sources
        sources = self._search_sources(question, top_k=5)
        context_block, citations = self._compose_context(question, sources)

        # If no LLM, provide a basic fallback using retrieved titles
        if not self.model_manager or not hasattr(self.model_manager, 'generate_text'):
            titles = ', '.join([c['title'] for c in citations[:3]]) if citations else ''
            fallback = "I cannot generate an AI answer right now. "
            if titles:
                fallback += f"Here are related sources: {titles}."
            return {'answer': fallback.strip(), 'citations': citations, 'used_context': context_block}

        # Build prompt and generate
        prompt = self._build_prompt(question, chat_history, context_block, user_profile)
        try:
            answer = self.model_manager.generate_text(prompt, max_length=500)
            return {'answer': (answer or '').strip(), 'citations': citations, 'used_context': context_block}
        except Exception as e:
            titles = ', '.join([c['title'] for c in citations[:3]]) if citations else ''
            fallback = f"Error generating answer: {e}" + (f". Related sources: {titles}" if titles else '')
            return {'answer': fallback.strip(), 'citations': citations, 'used_context': context_block}
