"""PDF Analyzer

This module provides PDF analysis utilities including text extraction,
storage into a vector database and Q&A. It performs lazy/safe imports so the
module can be instantiated even if optional dependencies (LLM, vector DB,
PyMuPDF) are not installed.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import json
import logging

logger = logging.getLogger(__name__)

# Try to import optional dependencies lazily and set availability flags
try:
    from utils.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except Exception:
    ModelManager = None
    MODEL_MANAGER_AVAILABLE = False

try:
    from utils.vector_db_manager import VectorDBManager, is_vector_db_available
    VECTOR_DB_AVAILABLE = True
except Exception:
    VectorDBManager = None
    is_vector_db_available = lambda: False
    VECTOR_DB_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except Exception:
    fitz = None
    FITZ_AVAILABLE = False


class PDFAnalyzer:
    """Analyzes PDF documents with vector database storage and AI-powered features.

    The class is resilient to missing optional dependencies. If an LLM or
    vector database is not available it will gracefully fall back to
    rule-based behavior for Q&A and skip storage operations.
    """

    def __init__(self):
        # Initialize model manager if available
        if MODEL_MANAGER_AVAILABLE and ModelManager is not None:
            try:
                self.model_manager = ModelManager()
            except Exception as e:
                logger.warning("Failed to initialize ModelManager: %s. LLM features will be limited.", e)
                self.model_manager = None
        else:
            self.model_manager = None
            logger.info("ModelManager not available. LLM features disabled.")

        # Initialize vector DB manager if available and initialized
        if VECTOR_DB_AVAILABLE and is_vector_db_available():
            try:
                self.vector_db_manager = VectorDBManager()
                self.vector_db = getattr(self.vector_db_manager, 'vector_db', None)
                logger.info("VectorDBManager initialized for PDF Analyzer")
            except Exception as e:
                logger.warning("Failed to initialize VectorDBManager: %s. Vector DB features will be limited.", e)
                self.vector_db_manager = None
                self.vector_db = None
        else:
            self.vector_db_manager = None
            self.vector_db = None
            logger.info("VectorDBManager not available. Vector DB features disabled.")

        # Template cache
        self._template_cache: Dict[str, str] = {}

    def _get_template_text(self, template_for: str) -> str:
        """Fetch a template text from vector DB by feature key, or empty string."""
        try:
            if template_for in self._template_cache:
                return self._template_cache[template_for]
            if self.vector_db_manager and hasattr(self.vector_db_manager, 'search_templates'):
                results = self.vector_db_manager.search_templates(template_for, top_k=1)
                if results:
                    md = results[0].get('metadata', {})
                    text = (md.get('template') or md.get('excerpt') or '').strip()
                    self._template_cache[template_for] = text
                    return text
        except Exception:
            pass
        return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from the PDF
        """
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF text extraction. Please install with: pip install PyMuPDF")
        
        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()
            
            # Close the document
            pdf_document.close()
            
            # Clean the extracted text
            text = self._clean_text(text)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def store_pdf_content(self, file_path: str, pdf_text: str, metadata: Dict[str, Any] = None) -> str:
        """Store PDF content in vector database
        
        Args:
            file_path (str): Path to the PDF file
            pdf_text (str): Extracted text from the PDF
            metadata (Dict[str, Any]): Additional metadata about the PDF
            
        Returns:
            str: Document ID of the stored PDF content
        """
        if not self.vector_db_manager:
            print("Warning: Vector database not available for storing PDF content")
            return None
            
        try:
            # Create metadata with PDF information
            pdf_metadata = {
                'type': 'pdf_content',
                'file_name': Path(file_path).name,
                'file_path': file_path,
                'word_count': len(pdf_text.split()),
                'character_count': len(pdf_text),
                'extraction_date': datetime.now().isoformat()
            }
            
            # Add an excerpt so we can use it later for RAG prompts
            try:
                cleaned_text = self._clean_text(pdf_text)
                # Build a concise excerpt using first meaningful paragraphs
                paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 60]
                excerpt_source = ' '.join(paragraphs[:2]) if paragraphs else cleaned_text
                pdf_metadata['excerpt'] = excerpt_source[:800]
            except Exception:
                pdf_metadata['excerpt'] = self._clean_text(pdf_text[:800])

            # Add any additional metadata
            if metadata:
                pdf_metadata.update(metadata)
            
            # Add PDF content to vector database
            doc_id = self.vector_db_manager.add_knowledge_item(pdf_text, pdf_metadata)
            print(f"✅ PDF content stored in vector database with ID: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"Error storing PDF content in vector database: {e}")
            return None
    
    def search_pdf_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search PDF content in vector database
        
        Args:
            query (str): Query text to search for
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores and metadata
        """
        if not self.vector_db_manager:
            print("Warning: Vector database not available for searching PDF content")
            return []
            
        try:
            # Search for PDF content
            results = self.vector_db_manager.search_knowledge(query, top_k)
            # Filter for PDF content only
            pdf_results = [r for r in results if r.get('metadata', {}).get('type') == 'pdf_content']
            print(f"✅ Found {len(pdf_results)} PDF content results")
            return pdf_results
        except Exception as e:
            print(f"Error searching PDF content in vector database: {e}")
            return []
    
    def retrieve_relevant_content(self, query: str, pdf_text: str, max_length: int = 1500) -> str:
        """Retrieve relevant content from PDF text based on query using improved method
        
        Args:
            query (str): Query to find relevant content for
            pdf_text (str): Full PDF text content
            max_length (int): Maximum length of retrieved content
            
        Returns:
            str: Relevant content from PDF text
        """
        try:
            # Split text into paragraphs for better context
            paragraphs = self._split_text_into_paragraphs(pdf_text)
            
            # Score paragraphs based on relevance to query
            relevant_paragraphs = []
            query_words = query.lower().split()
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                # Calculate relevance score based on word matching
                score = 0
                paragraph_lower = paragraph.lower()
                
                # Check for exact query matches
                if query.lower() in paragraph_lower:
                    score += 10
                
                # Check for word matches
                for word in query_words:
                    if word in paragraph_lower:
                        score += 2
                
                # Bonus for longer paragraphs with matches (more context)
                if score > 0:
                    score += min(len(paragraph) / 100, 5)
                
                if score > 0:
                    relevant_paragraphs.append((score, paragraph))
            
            # Sort by relevance and select top paragraphs
            relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)
            
            # Build result within max_length constraint
            result = ""
            for score, paragraph in relevant_paragraphs[:5]:  # Top 5 paragraphs
                if len(result) + len(paragraph) + 2 <= max_length:
                    result += paragraph + "\n\n"
                else:
                    # Add partial content if we can
                    remaining_chars = max_length - len(result) - 2
                    if remaining_chars > 50:  # Only add if we have meaningful space
                        result += paragraph[:remaining_chars] + "\n\n"
                    break
            
            # If no relevant content found, return beginning of document
            if not result.strip():
                result = pdf_text[:max_length]
            
            # Clean the result before returning
            result = self._clean_text(result)
            return result.strip()
        except Exception as e:
            print(f"Error retrieving relevant content: {e}")
            # Fallback to first part of document
            fallback = pdf_text[:max_length]
            return self._clean_text(fallback)
    
    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs for better processing"""
        # Split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        
        # Further split large paragraphs
        result = []
        for paragraph in paragraphs:
            if len(paragraph) > 1000:  # If paragraph is very long
                # Split by single newlines
                lines = paragraph.split('\n')
                current_para = ""
                for line in lines:
                    if len(current_para) + len(line) < 800:
                        current_para += line + "\n"
                    else:
                        if current_para.strip():
                            result.append(current_para.strip())
                        current_para = line + "\n"
                if current_para.strip():
                    result.append(current_para.strip())
            else:
                if paragraph.strip():
                    result.append(paragraph.strip())
        
        return result
    
    def summarize_pdf(self, pdf_text: str, max_length: int = 300) -> str:
        """Generate a comprehensive summary of the PDF content using LLM
        
        Args:
            pdf_text (str): Text content of the PDF
            max_length (int): Maximum length of the summary
            
        Returns:
            str: Comprehensive summary of the PDF content
        """
        if not self.model_manager or not hasattr(self.model_manager, 'generate_text'):
            return "Summary not available: Language model not available."
            
        try:
            # Clean the input text
            pdf_text = self._clean_text(pdf_text)
            
            # Use a more sophisticated approach with LLM to generate comprehensive summary
            # First, try to extract key sections if this looks like a structured document
            lines = pdf_text.split('\n')
            executive_summary = ""
            found_summary = False
            
            # Look for executive summary or summary sections
            for line in lines:
                if 'executive summary' in line.lower() or 'summary' in line.lower():
                    found_summary = True
                    continue
                if found_summary and line.strip() and len(line.strip()) > 20:
                    executive_summary += line.strip() + " "
                    if len(executive_summary) > 500:  # Increase limit for better content
                        break
                if found_summary and line.strip() and len(executive_summary) > 100 and len(line.strip()) < 20:
                    # Likely end of summary section
                    break
            
            if len(executive_summary) > 100:
                clean_summary = self._clean_text(executive_summary[:max_length].strip())
                if not self._is_garbled(clean_summary) and len(clean_summary) > 30:
                    return clean_summary
            
            # Enhanced LLM-based summarization with better prompt engineering
            template_instructions = self._get_template_text('pdf_summary')
            # Use chunks of the document for better context
            chunk_size = 3000  # Increased chunk size for better context
            if len(pdf_text) <= chunk_size:
                # For shorter documents, use the entire text
                content_to_summarize = pdf_text
            else:
                # For longer documents, use a more intelligent approach
                # Take the beginning, middle, and end sections for comprehensive coverage
                start_chunk = pdf_text[:chunk_size//3]
                middle_chunk = pdf_text[len(pdf_text)//2 - chunk_size//6:len(pdf_text)//2 + chunk_size//6]
                end_chunk = pdf_text[-chunk_size//3:] if len(pdf_text) > chunk_size//3 else pdf_text[len(pdf_text)//2:]
                content_to_summarize = start_chunk + "\n...\n" + middle_chunk + "\n...\n" + end_chunk
            
            # Enhanced prompt for comprehensive summarization
            prompt = f"""{template_instructions}
Please provide a comprehensive summary of the following document. 
Your summary should include:
1. The main topic or purpose of the document
2. Key points and important information
3. Main findings or conclusions (if applicable)
4. Any significant data or statistics mentioned
5. The overall tone or intent of the document

Document content:
{content_to_summarize}

Please provide a well-structured summary in one paragraph that captures the essence of the document.
Summary:"""
            
            summary = self.model_manager.generate_text(prompt, max_length=max_length + 100)
            
            # Post-processing to clean up the response
            summary = self._clean_text(summary.strip())
            
            # Remove common artifacts
            if summary.lower().startswith("summary:"):
                summary = summary[8:].strip()
            elif summary.lower().startswith("here is a summary"):
                parts = summary.split(":", 1)
                if len(parts) > 1:
                    summary = parts[1].strip()
            
            # If the summary is still too generic or garbled, try a different approach
            if (summary.lower().startswith('summarize') or 'document' in summary[:50].lower() or 
                len(summary) < 50 or self._is_garbled(summary)):
                # Extract key sentences that contain important information
                sentences = pdf_text.split('.')
                key_sentences = []
                # Look for sentences with important keywords
                important_keywords = [
                    'conclude', 'find', 'result', 'achieve', 'develop', 'implement', 
                    'create', 'design', 'build', 'improve', 'increase', 'decrease',
                    'purpose', 'objective', 'goal', 'aim', 'introduce', 'present',
                    'study', 'research', 'analysis', 'method', 'approach', 'solution'
                ]
                
                for sentence in sentences:
                    sentence = self._clean_text(sentence.strip())
                    if (any(keyword in sentence.lower() for keyword in important_keywords) and 
                        len(sentence) > 30 and not self._is_garbled(sentence)):
                        key_sentences.append(sentence)
                        if len(' '.join(key_sentences)) > max_length:
                            break
                
                if key_sentences:
                    # Join key sentences and trim to max_length
                    summary = '. '.join(key_sentences) + '.'
                else:
                    # Last resort - take a representative sample from different parts of the document
                    paragraphs = pdf_text.split('\n\n')
                    # Filter out very short paragraphs
                    meaningful_paragraphs = [p for p in paragraphs if len(self._clean_text(p).strip()) > 50]
                    if meaningful_paragraphs:
                        # Take a few representative paragraphs
                        sample_paragraphs = meaningful_paragraphs[:min(3, len(meaningful_paragraphs))]
                        sample_text = '\n\n'.join(sample_paragraphs)
                        # Create a simpler prompt for this sample
                        prompt = f"Summarize the following text in one paragraph:\n\n{sample_text[:1000]}"
                        summary = self.model_manager.generate_text(prompt, max_length=max_length + 50)
                        summary = self._clean_text(summary.strip())
                        if summary.lower().startswith("summary:"):
                            summary = summary[8:].strip()
                    else:
                        # Final fallback - take first part of document
                        clean_text = self._clean_text(pdf_text[:max_length])
                        summary = clean_text.split('\n\n')[0] + "..." if '\n\n' in clean_text else clean_text[:max_length-3] + "..."
            
            # Ensure we don't exceed max_length and clean final output
            clean_summary = self._clean_text(summary.strip())
            if len(clean_summary) > max_length:
                # Try to find a good breaking point
                if '. ' in clean_summary[:max_length]:
                    clean_summary = '. '.join(clean_summary[:max_length].split('. ')[:-1]) + "."
                else:
                    clean_summary = clean_summary[:max_length-3] + "..."
            
            # If still garbled, provide a simple fallback
            if self._is_garbled(clean_summary):
                # Extract first few meaningful sentences
                sentences = pdf_text.split('.')
                clean_sentences = []
                for sentence in sentences[:10]:
                    clean_sentence = self._clean_text(sentence.strip())
                    if len(clean_sentence) > 20 and not self._is_garbled(clean_sentence):
                        clean_sentences.append(clean_sentence)
                        if len(' '.join(clean_sentences)) > max_length // 2:
                            break
                
                if clean_sentences:
                    clean_summary = '. '.join(clean_sentences) + '.'
                else:
                    clean_summary = "Document summary could not be generated due to content formatting issues."
            
            return clean_summary.strip() if clean_summary else "Document summary could not be generated."
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def answer_question(self, pdf_text: str, question: str, max_length: int = 300) -> str:
        """Answer a question about the PDF content using improved method with better context retrieval
        
        Args:
            pdf_text (str): Text content of the PDF
            question (str): Question to answer
            max_length (int): Maximum length of the answer
            
        Returns:
            str: Answer to the question
        """
        if not self.model_manager or not hasattr(self.model_manager, 'generate_text'):
            # Fallback to rule-based approach if LLM not available
            return self._rule_based_answer(pdf_text, question, max_length)
            
        try:
            # Build enhanced RAG context: current PDF context + similar items from vector DB (knowledge/PDF excerpts)
            relevant_content = self._build_rag_context(question, pdf_text, max_length=2000)
            
            # Clean the relevant content to remove any encoding issues
            relevant_content = self._clean_text(relevant_content)
            
            # Enhanced prompt with better structure and instructions
            template_instructions = self._get_template_text('pdf_qa')
            prompt = f"""{template_instructions}
You are an expert document analyzer. Based on the following retrieved context, please answer the question accurately and concisely.

Retrieved Context:
{relevant_content}

Question: {question}

Instructions:
1. Provide a clear and direct answer based ONLY on the information in the document excerpt
2. If the information is not available in the document, respond with: "I cannot find specific information about this question in the document."
3. If the question is unclear or ambiguous, ask for clarification
4. Keep your answer focused and relevant to the question
5. If applicable, include specific data, facts, or quotes from the document

Answer:"""
            
            answer = self.model_manager.generate_text(prompt, max_length=max_length + 150)
            
            # Clean up the answer to handle encoding issues
            answer = self._clean_text(answer.strip())
            
            # Remove common prompt artifacts
            if answer.lower().startswith(("answer:", "response:", "reply:")):
                # Find the first colon and take everything after it
                parts = answer.split(":", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
            
            # Handle cases where the model repeats the question or provides generic responses
            if (answer.lower().startswith("question:") or 
                "the question" in answer.lower()[:100] or
                answer.lower().startswith("based on the") or
                len(answer) < 15):
                # Try with a more focused approach
                focused_answer = self._focused_qa_approach(pdf_text, question, max_length)
                if len(focused_answer) > len(answer) and not focused_answer.startswith("Error"):
                    answer = focused_answer
            
            # Ensure we're not returning the prompt itself or generic responses
            if (question.lower() in answer.lower()[:len(question) + 30] and len(answer) < len(question) + 100) or \
               answer.lower().startswith("i cannot answer") or \
               ("i cannot" in answer.lower() and "document" in answer.lower()):
                # Try rule-based approach as fallback
                rule_based_answer = self._rule_based_answer(pdf_text, question, max_length)
                # Only use rule-based if it's more substantive
                if len(rule_based_answer) > 20 and not rule_based_answer.startswith("Error"):
                    answer = rule_based_answer
            
            # Final cleanup and validation
            answer = self._clean_text(answer.strip())
            
            if len(answer) > max_length:
                # Try to find a good breaking point
                if '. ' in answer[:max_length]:
                    answer = '. '.join(answer[:max_length].split('. ')[:-1]) + "."
                else:
                    answer = answer[:max_length-3] + "..."
            
            # If answer is still garbled or too short, use rule-based approach
            if len(answer) < 10 or self._is_garbled(answer):
                rule_based_answer = self._rule_based_answer(pdf_text, question, max_length)
                if len(rule_based_answer) > 20 and not rule_based_answer.startswith("Error"):
                    return rule_based_answer
            
            # If we still don't have a good answer, provide a meaningful fallback
            if len(answer) < 20 or "document" in answer.lower() or self._is_garbled(answer):
                return "I cannot find specific information about this question in the document. Please try rephrasing your question or ask about specific topics mentioned in the document."
            
            return answer.strip() if answer else "I cannot find specific information about this question in the document."
        except Exception as e:
            # Fallback to rule-based approach on error
            print(f"Error in LLM-based answering: {e}")
            return self._rule_based_answer(pdf_text, question, max_length)

    def _build_rag_context(self, question: str, pdf_text: str, max_length: int = 2000) -> str:
        """Compose an augmented context using current document content and similar items from vector DB.

        Returns a concatenated string trimmed to max_length characters.
        """
        try:
            parts: List[str] = []
            # 1) Current PDF relevant content
            current_doc_ctx = self._retrieve_enhanced_context(question, pdf_text, max_length=max_length // 2)
            if current_doc_ctx:
                parts.append("[Current Document]\n" + current_doc_ctx)

            # 2) Similar knowledge items (if vector DB available)
            if self.vector_db_manager:
                try:
                    kb_results = self.vector_db_manager.search_knowledge(question, top_k=3)
                    kb_snippets = []
                    for r in kb_results:
                        md = r.get('metadata', {})
                        title = md.get('title') or md.get('file_name') or 'Knowledge Item'
                        excerpt = md.get('excerpt') or ''
                        if excerpt:
                            kb_snippets.append(f"- {title}: {excerpt[:400]}")
                    if kb_snippets:
                        parts.append("[Related Knowledge]\n" + "\n".join(kb_snippets))
                except Exception:
                    pass

                # 3) Similar PDF items (stored content)
                try:
                    pdf_sim = self.search_pdf_content(question, top_k=2)
                    pdf_snippets = []
                    for r in pdf_sim:
                        md = r.get('metadata', {})
                        title = md.get('title') or md.get('file_name') or 'PDF Document'
                        excerpt = md.get('excerpt') or ''
                        if excerpt:
                            pdf_snippets.append(f"- {title}: {excerpt[:400]}")
                    if pdf_snippets:
                        parts.append("[Similar Documents]\n" + "\n".join(pdf_snippets))
                except Exception:
                    pass

            # Join and trim
            composed = "\n\n".join(parts).strip()
            return composed[:max_length] if composed else self._retrieve_enhanced_context(question, pdf_text, max_length=max_length)
        except Exception:
            # Fallback to current document context
            return self._retrieve_enhanced_context(question, pdf_text, max_length=max_length)
    
    def _retrieve_enhanced_context(self, question: str, pdf_text: str, max_length: int = 2000) -> str:
        """Enhanced context retrieval with better relevance scoring and chunking"""
        try:
            # Split text into paragraphs for better context
            paragraphs = self._split_text_into_paragraphs(pdf_text)
            
            # Score paragraphs based on relevance to question
            relevant_paragraphs = []
            question_words = [word.lower() for word in question.split() if len(word) > 2]
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                # Calculate relevance score based on word matching and semantic similarity
                score = 0
                paragraph_lower = paragraph.lower()
                
                # Exact question matches (high weight)
                if question.lower() in paragraph_lower:
                    score += 20
                
                # Word matches (medium weight)
                for word in question_words:
                    if word in paragraph_lower:
                        score += 3
                
                # Semantic keywords (lower weight)
                semantic_keywords = [
                    'purpose', 'objective', 'goal', 'aim', 'introduce', 'present',
                    'conclude', 'find', 'result', 'achieve', 'develop', 'implement', 
                    'create', 'design', 'build', 'improve', 'increase', 'decrease',
                    'study', 'research', 'analysis', 'method', 'approach', 'solution',
                    'problem', 'challenge', 'issue', 'benefit', 'advantage', 'disadvantage'
                ]
                
                for keyword in semantic_keywords:
                    if keyword in paragraph_lower:
                        score += 1
                
                # Bonus for longer paragraphs with matches (more context)
                if score > 0:
                    score += min(len(paragraph) / 200, 3)
                
                if score > 0:
                    relevant_paragraphs.append((score, paragraph))
            
            # Sort by relevance and select top paragraphs
            relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)
            
            # Build result within max_length constraint
            result = ""
            for score, paragraph in relevant_paragraphs[:8]:  # Increased from 5 to 8
                if len(result) + len(paragraph) + 2 <= max_length:
                    result += paragraph + "\n\n"
                else:
                    # Add partial content if we can
                    remaining_chars = max_length - len(result) - 2
                    if remaining_chars > 100:  # Only add if we have meaningful space
                        result += paragraph[:remaining_chars] + "\n\n"
                    break
            
            # If no relevant content found, return strategic parts of document
            if not result.strip():
                # Take beginning, middle, and end sections
                doc_length = len(pdf_text)
                if doc_length <= max_length:
                    result = pdf_text
                else:
                    start_section = pdf_text[:max_length//3]
                    middle_section = pdf_text[doc_length//2 - max_length//6:doc_length//2 + max_length//6]
                    end_section = pdf_text[-max_length//3:] if doc_length > max_length//3 else pdf_text[doc_length//2:]
                    result = f"{start_section}\n...\n{middle_section}\n...\n{end_section}"
            
            # Clean the result before returning
            result = self._clean_text(result)
            return result.strip()
        except Exception as e:
            print(f"Error in enhanced context retrieval: {e}")
            # Fallback to simpler approach
            fallback = pdf_text[:max_length]
            return self._clean_text(fallback)
    
    def _focused_qa_approach(self, pdf_text: str, question: str, max_length: int = 300) -> str:
        """More focused QA approach for better accuracy"""
        try:
            # Get relevant content with higher precision
            relevant_content = self._retrieve_enhanced_context(question, pdf_text, max_length=1500)
            
            # Use a more structured prompt
            prompt = f"""Document Analysis Task:
1. Read the following document excerpt carefully
2. Identify the specific information requested in the question
3. Extract only the relevant facts from the document
4. Provide a concise, factual answer

Document Excerpt:
{relevant_content}

Question: {question}

Answer (be concise and factual):"""
            
            answer = self.model_manager.generate_text(prompt, max_length=max_length + 100)
            answer = answer.strip()
            
            # Clean up the answer
            answer = self._clean_text(answer)
            
            if answer.lower().startswith(("answer:", "response:", "reply:")):
                parts = answer.split(":", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
            
            # If still not good, try with bullet points format
            if len(answer) < 20 or "document" in answer.lower():
                prompt = f"""Extract facts from the document to answer: {question}

Document:
{relevant_content}

Facts only:"""
                
                answer = self.model_manager.generate_text(prompt, max_length=max_length + 50)
                answer = answer.strip()
                answer = self._clean_text(answer)
            
            return answer[:max_length].strip() if len(answer) > max_length else answer.strip()
        except Exception as e:
            print(f"Error in focused QA approach: {e}")
            return self._rule_based_answer(pdf_text, question, max_length)
    
    def _rule_based_answer(self, pdf_text: str, question: str, max_length: int = 300) -> str:
        """Rule-based approach to answer questions when LLM fails
        
        Args:
            pdf_text (str): Text content of the PDF
            question (str): Question to answer
            max_length (int): Maximum length of the answer
            
        Returns:
            str: Answer to the question
        """
        try:
            # Clean the input text
            pdf_text = self._clean_text(pdf_text)
            question = self._clean_text(question)
            
            # Split into sentences
            sentences = [self._clean_text(s.strip()) for s in pdf_text.split('.') if s.strip()]
            question_words = [word.lower() for word in question.split() if len(word) > 3]
            
            # Look for sentences that contain question keywords
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip garbled sentences
                if self._is_garbled(sentence) or len(sentence) < 10:
                    continue
                    
                if any(word in sentence_lower for word in question_words) and len(sentence.strip()) > 10:
                    relevant_sentences.append(sentence.strip() + ".")
                    if len(" ".join(relevant_sentences)) > max_length // 2:
                        break
            
            if relevant_sentences:
                answer = " ".join(relevant_sentences)
                clean_answer = self._clean_text(answer.strip())
                # Validate the answer is not garbled
                if not self._is_garbled(clean_answer) and len(clean_answer) > 20:
                    return clean_answer
            
            # If no keyword matches, try to find sentences with numbers/data for financial questions
            financial_keywords = ['revenue', 'profit', 'financial', 'money', 'dollars', 'income', 'growth', 'increase', 'decrease', 'budget', 'cost', 'price', 'salary']
            if any(word in question.lower() for word in financial_keywords):
                # Look for sentences with numbers and financial terms
                financial_sentences = []
                for sentence in sentences:
                    sentence_clean = self._clean_text(sentence.strip())
                    # Skip garbled sentences
                    if self._is_garbled(sentence_clean) or len(sentence_clean) < 10:
                        continue
                        
                    if any(char.isdigit() for char in sentence_clean) and len(sentence_clean) > 10:
                        if any(term in sentence_clean.lower() for term in financial_keywords):
                            financial_sentences.append(sentence_clean + ".")
                            if len(" ".join(financial_sentences)) > max_length // 2:
                                break
                
                if financial_sentences:
                    answer = " ".join(financial_sentences)
                    clean_answer = self._clean_text(answer.strip())
                    if not self._is_garbled(clean_answer):
                        return clean_answer
                else:
                    return "I cannot find specific financial information about this in the document."
            
            # Try to find sentences with named entities (names, places, organizations)
            name_keywords = ['company', 'organization', 'corporation', 'inc', 'llc', 'ltd', 'firm', 'enterprise', 'business']
            if any(word in question.lower() for word in name_keywords):
                # Look for capitalized phrases that might be company names
                name_sentences = []
                for sentence in sentences[:50]:  # Check first 50 sentences
                    sentence_clean = self._clean_text(sentence.strip())
                    # Skip garbled sentences
                    if self._is_garbled(sentence_clean) or len(sentence_clean) < 10:
                        continue
                        
                    # Look for capitalized words that might be names
                    words = sentence_clean.split()
                    capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
                    if len(capitalized_words) >= 2 and len(sentence_clean) > 10:
                        name_sentences.append(sentence_clean + ".")
                        if len(" ".join(name_sentences)) > max_length // 2:
                            break
                
                if name_sentences:
                    answer = " ".join(name_sentences)
                    clean_answer = self._clean_text(answer.strip())
                    if not self._is_garbled(clean_answer):
                        return clean_answer
                else:
                    return "I cannot find specific information about organizations or companies in the document."
            
            # Generic fallback - look for sentences with purpose-related keywords
            purpose_keywords = ['purpose', 'objective', 'goal', 'aim', 'introduce', 'present', 'conclude', 'find', 'result']
            if any(word in question.lower() for word in ['purpose', 'main', 'primary']):
                purpose_sentences = []
                for sentence in sentences[:30]:
                    sentence_clean = self._clean_text(sentence.strip())
                    # Skip garbled sentences
                    if self._is_garbled(sentence_clean) or len(sentence_clean) < 10:
                        continue
                        
                    if any(keyword in sentence_clean.lower() for keyword in purpose_keywords):
                        purpose_sentences.append(sentence_clean + ".")
                        if len(" ".join(purpose_sentences)) > max_length // 2:
                            break
                
                if purpose_sentences:
                    answer = " ".join(purpose_sentences)
                    clean_answer = self._clean_text(answer.strip())
                    if not self._is_garbled(clean_answer):
                        return clean_answer
            
            # Last resort - provide a general response
            return "I cannot find specific information about this question in the document. Please try rephrasing your question or ask about specific topics mentioned in the document."
            
        except Exception as e:
            return f"Error in rule-based answering: {str(e)}"
    
    def analyze_pdf(self, file_path: str) -> Dict[str, Any]:
        """Complete analysis of a PDF document
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Complete analysis results including text, summary, and metadata
        """
        try:
            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(file_path)
            
            # Generate summary
            summary = self.summarize_pdf(pdf_text)
            
            # Store in vector database
            doc_id = self.store_pdf_content(file_path, pdf_text)
            
            # Extract key information
            key_info = self.extract_key_information(pdf_text)
            
            # Analyze document structure
            structure_analysis = self.analyze_document_structure(pdf_text)
            
            # Extract document sentiment
            sentiment_analysis = self.analyze_sentiment(pdf_text)
            
            # Extract named entities
            named_entities = self.extract_named_entities(pdf_text)
            
            # Prepare result
            result = {
                'text': pdf_text,
                'summary': summary,
                'word_count': len(pdf_text.split()),
                'character_count': len(pdf_text),
                'document_id': doc_id,
                'file_name': Path(file_path).name,
                'analysis_date': datetime.now().isoformat(),
                'key_information': key_info,
                'structure_analysis': structure_analysis,
                'sentiment_analysis': sentiment_analysis,
                'named_entities': named_entities
            }
            
            return result
        except Exception as e:
            raise Exception(f"Error analyzing PDF: {str(e)}")
    
    def extract_key_information(self, pdf_text: str) -> Dict[str, Any]:
        """Extract key information from PDF text
        
        Args:
            pdf_text (str): Text content of the PDF
            
        Returns:
            Dict[str, Any]: Extracted key information
        """
        # Extract potential titles (sentences that look like headings)
        lines = pdf_text.split('\n')
        titles = [line.strip() for line in lines if line.strip() and 
                  len(line.strip()) > 3 and len(line.strip()) < 100 and
                  (line.isupper() or (line.istitle() and len(line.split()) < 10))]
        
        # Extract potential dates (simple pattern matching)
        import re
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b'
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, pdf_text, re.IGNORECASE))
        
        # Extract potential emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, pdf_text)
        
        # Extract potential phone numbers
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, pdf_text)
        
        # Extract potential URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, pdf_text)
        
        # Extract potential names (simple pattern - capitalized words)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        names = re.findall(name_pattern, pdf_text)
        
        # Extract potential organizations (all caps phrases with 2+ words)
        org_pattern = r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b'
        organizations = re.findall(org_pattern, pdf_text)
        
        return {
            'titles': titles[:10],  # Limit to first 10 titles
            'dates': list(set(dates)),  # Remove duplicates
            'emails': list(set(emails)),
            'phones': list(set(phones)),
            'urls': list(set(urls)),
            'names': list(set(names))[:20],  # Limit to first 20 names
            'organizations': list(set(organizations))[:10]  # Limit to first 10 organizations
        }
    
    def analyze_document_structure(self, pdf_text: str) -> Dict[str, Any]:
        """Analyze the structure of the PDF document
        
        Args:
            pdf_text (str): Text content of the PDF
            
        Returns:
            Dict[str, Any]: Document structure analysis
        """
        lines = pdf_text.split('\n')
        
        # Count different types of lines
        empty_lines = sum(1 for line in lines if not line.strip())
        short_lines = sum(1 for line in lines if 1 <= len(line.strip()) <= 10)
        medium_lines = sum(1 for line in lines if 11 <= len(line.strip()) <= 50)
        long_lines = sum(1 for line in lines if len(line.strip()) > 50)
        
        # Estimate sections based on line length patterns
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(stripped)
        
        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Analyze bullet points and lists
        bullet_points = [line for line in lines if line.strip().startswith(('-', '*', '•', '·'))]
        
        # Analyze numbered lists
        numbered_lines = [line for line in lines if re.match(r'^\s*\d+[\.\)]', line)]
        
        return {
            'total_lines': len(lines),
            'empty_lines': empty_lines,
            'short_lines': short_lines,
            'medium_lines': medium_lines,
            'long_lines': long_lines,
            'estimated_paragraphs': len(paragraphs),
            'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            'bullet_points': len(bullet_points),
            'numbered_lines': len(numbered_lines)
        }
    
    def analyze_sentiment(self, pdf_text: str) -> Dict[str, Any]:
        """Analyze the sentiment of the PDF document
        
        Args:
            pdf_text (str): Text content of the PDF
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        if not self.model_manager or not hasattr(self.model_manager, 'analyze_sentiment'):
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            # Analyze sentiment of the first 1000 characters for performance
            sentiment_scores = self.model_manager.analyze_sentiment(pdf_text[:1000])
            return {
                "sentiment_scores": sentiment_scores,
                "dominant_sentiment": max(sentiment_scores, key=sentiment_scores.get) if sentiment_scores else "neutral",
                "confidence": max(sentiment_scores.values()) if sentiment_scores else 0.5
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def extract_named_entities(self, pdf_text: str) -> List[Dict[str, Any]]:
        """Extract named entities from the PDF document
        
        Args:
            pdf_text (str): Text content of the PDF
            
        Returns:
            List[Dict[str, Any]]: List of extracted named entities
        """
        if not self.model_manager or not hasattr(self.model_manager, 'extract_entities'):
            return []
        
        try:
            # Extract entities from the first 2000 characters for performance
            entities = self.model_manager.extract_entities(pdf_text[:2000])
            return entities
        except Exception as e:
            print(f"Error extracting named entities: {e}")
            return []
    
    def extract_tables_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF if available
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, Any]]: List of extracted tables with their data
        """
        try:
            if not FITZ_AVAILABLE:
                return []
            
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            tables = []
            
            # Try to extract tables from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Look for table-like structures
                # This is a simplified approach - in a real implementation, 
                # you might use specialized libraries like camelot or pdfplumber
                pass  # Implementation would go here
            
            # Close the document
            pdf_document.close()
            return tables
        except Exception as e:
            print(f"Warning: Could not extract tables from PDF: {e}")
            return []
    
    def find_similar_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents in the vector database
        
        Args:
            query_text (str): Text to search for similar documents
            top_k (int): Number of similar documents to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents
        """
        if not self.vector_db_manager:
            return []
        
        try:
            # Search for similar documents
            results = self.vector_db_manager.search_knowledge(query_text, top_k)
            # Filter for PDF content only
            pdf_results = [r for r in results if r.get('metadata', {}).get('type') == 'pdf_content']
            return pdf_results
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text to remove encoding issues and garbled characters"""
        if not text:
            return ""
        
        # Remove common garbled characters and encoding issues
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '').replace('\ufffd', '').replace('Â', '').replace('', '')
        
        # Remove excessive special characters that are likely encoding issues
        import re
        # Remove sequences of special characters that are likely encoding issues
        text = re.sub(r'[^\x00-\x7F]*[À-ÿ]+[^\x00-\x7F]*', ' ', text)
        
        # Remove isolated special characters that are likely encoding artifacts
        text = re.sub(r'[\x80-\xFF]', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_garbled(self, text: str) -> bool:
        """Check if text appears to be garbled or contains encoding issues"""
        if not text:
            return True
        
        # Check for excessive special characters that indicate encoding issues
        special_char_count = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0 and special_char_count / len(text) > 0.3:
            return True
        
        # Check for common garbled patterns
        garbled_patterns = ['Â', '', 'ï¿½', '\ufffd']
        if any(pattern in text for pattern in garbled_patterns):
            return True
        
        # Check if the text has too many non-printable characters
        non_printable_count = sum(1 for c in text if not c.isprintable() and not c.isspace())
        if len(text) > 0 and non_printable_count / len(text) > 0.1:
            return True
        
        return False