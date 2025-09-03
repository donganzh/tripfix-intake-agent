import PyPDF2
import os
from typing import List, Dict
import re

class PDFProcessor:
    def __init__(self, data_folder: str = "data/regulations"):
        self.data_folder = data_folder
        self.ensure_folder_exists()
    
    def ensure_folder_exists(self):
        """Ensure the data folder exists"""
        os.makedirs(self.data_folder, exist_ok=True)
    
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
        """Split text into overlapping chunks for vector storage"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Clean up text
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            
            if len(chunk_text) > 50:  # Only include substantial chunks
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'chunk_id': len(chunks),
                        'start_word': i,
                        'word_count': len(chunk_words)
                    }
                })
        
        return chunks
    
    def process_all_pdfs(self) -> Dict[str, List[Dict]]:
        """Process all PDFs in the data folder"""
        processed_docs = {}
        
        for filename in os.listdir(self.data_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.data_folder, filename)
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    chunks = self.chunk_text(text)
                    # Add source metadata to each chunk
                    for chunk in chunks:
                        chunk['metadata']['source'] = filename
                        chunk['metadata']['regulation_type'] = self.detect_regulation_type(filename, text)
                    
                    processed_docs[filename] = chunks
        
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