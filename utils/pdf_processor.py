import PyPDF2
import os
from typing import List, Dict, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import hashlib

class PDFProcessor:
    def __init__(self, data_folder: str = "data/regulations"):
        self.data_folder = data_folder
        self.ensure_folder_exists()
        self._download_nltk_data()
    
    def ensure_folder_exists(self):
        """Ensure the data folder exists"""
        os.makedirs(self.data_folder, exist_ok=True)
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def detect_content_type(self, text: str) -> str:
        """Detect the type of content (article, section, definition, etc.)"""
        text_lower = text.lower()
        
        # Check for article patterns
        if re.search(r'\barticle\s+\d+', text_lower):
            return 'article'
        elif re.search(r'\bsection\s+\d+', text_lower):
            return 'section'
        elif re.search(r'\bdefinition\b|\bmeans\b', text_lower):
            return 'definition'
        elif re.search(r'\bcompensation\b|\bentitlement\b', text_lower):
            return 'compensation'
        elif re.search(r'\bdelay\b|\bretard\b', text_lower):
            return 'delay_provision'
        elif re.search(r'\bexemption\b|\bexception\b', text_lower):
            return 'exemption'
        else:
            return 'general'
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for better searchability"""
        # Remove common words and extract meaningful terms
        stop_words = set(stopwords.words('english') + stopwords.words('french'))
        words = word_tokenize(text.lower())
        
        # Filter out stop words, numbers, and short words
        key_terms = [
            word for word in words 
            if word.isalpha() and len(word) > 3 and word not in stop_words
        ]
        
        # Return unique terms, limited to most relevant ones
        return list(set(key_terms))[:20]
    
    def chunk_text(self, text: str, target_chunk_size: int = 800, overlap_sentences: int = 2) -> List[Dict]:
        """Split text into semantically-aware chunks with better overlap"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences for better semantic boundaries
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed target size, create a chunk
            if current_size + sentence_words > target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Create chunk with enhanced metadata
                chunk_metadata = self._create_chunk_metadata(
                    chunk_text, chunk_id, i - len(current_chunk), 
                    len(current_chunk), text
                )
                
                chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # Start new chunk with overlap
                overlap_sentences_actual = min(overlap_sentences, len(current_chunk))
                current_chunk = current_chunk[-overlap_sentences_actual:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_words
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(
                chunk_text, chunk_id, len(sentences) - len(current_chunk),
                len(current_chunk), text
            )
            
            chunks.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_text: str, chunk_id: int, 
                             start_sentence: int, sentence_count: int, 
                             full_text: str) -> Dict:
        """Create enhanced metadata for a chunk"""
        # Generate content hash for deduplication
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        
        # Detect content type
        content_type = self.detect_content_type(chunk_text)
        
        # Extract key terms
        key_terms = self.extract_key_terms(chunk_text)
        
        # Calculate chunk position in document
        chunk_position = start_sentence / max(1, len(full_text.split('.')))
        
        return {
            'chunk_id': chunk_id,
            'content_hash': content_hash,
            'content_type': content_type,
            'start_sentence': start_sentence,
            'sentence_count': sentence_count,
            'word_count': len(chunk_text.split()),
            'char_count': len(chunk_text),
            'chunk_position': round(chunk_position, 3),
            'key_terms': '|'.join(key_terms[:10]),  # Convert list to pipe-separated string
            'key_terms_count': len(key_terms),
            'has_compensation_info': 'compensation' in chunk_text.lower() or 'entitlement' in chunk_text.lower(),
            'has_delay_info': 'delay' in chunk_text.lower() or 'retard' in chunk_text.lower(),
            'has_exemption_info': 'exemption' in chunk_text.lower() or 'exception' in chunk_text.lower()
        }
    
    def process_all_pdfs(self) -> Dict[str, List[Dict]]:
        """Process all PDFs in the data folder with enhanced metadata"""
        processed_docs = {}
        
        for filename in os.listdir(self.data_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.data_folder, filename)
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    chunks = self.chunk_text(text)
                    
                    # Add document-level metadata to each chunk
                    regulation_type = self.detect_regulation_type(filename, text)
                    document_key_terms = self.extract_key_terms(text)
                    
                    for chunk in chunks:
                        # Add source and document-level metadata
                        chunk['metadata'].update({
                            'source': filename,
                            'regulation_type': regulation_type,
                            'document_key_terms': '|'.join(document_key_terms[:15]),  # Convert to string
                            'document_key_terms_count': len(document_key_terms),
                            'total_document_chunks': len(chunks),
                            'document_word_count': len(text.split()),
                            'document_char_count': len(text)
                        })
                    
                    processed_docs[filename] = chunks
                    print(f"Processed {filename}: {len(chunks)} chunks, {regulation_type} regulation")
        
        return processed_docs
    
    def detect_regulation_type(self, filename: str, text: str) -> str:
        """Detect if document is APPR, EU261, or other"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if 'appr' in filename_lower or 'canada' in filename_lower or 'canadian air passenger rights' in text_lower:
            return 'APPR'
        elif 'eu' in filename_lower or '261' in filename_lower or 'regulation (ec) no 261/2004' in text_lower:
            return 'EU261'
        else:
            return 'OTHER'