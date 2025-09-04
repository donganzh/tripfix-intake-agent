"""
File Processing Utility for TripFix
Handles uploaded supporting documents and extracts relevant information
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import mimetypes
import hashlib


class FileProcessor:
    """Processes uploaded files and extracts relevant information"""
    
    def __init__(self, upload_dir: str = "data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file types
        self.supported_types = {
            'application/pdf': self._process_pdf,
            'image/jpeg': self._process_image,
            'image/png': self._process_image,
            'image/jpg': self._process_image,
            'text/plain': self._process_text,
            'application/msword': self._process_doc,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx
        }
    
    def process_uploaded_file(self, file_content: bytes, filename: str, 
                            session_id: str) -> Dict[str, Any]:
        """Process an uploaded file and return metadata"""
        try:
            # Generate unique filename
            file_extension = Path(filename).suffix
            unique_filename = f"{session_id}_{uuid.uuid4().hex}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Get file info
            file_size = len(file_content)
            file_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            
            # Process file based on type
            extracted_text = ""
            metadata = {
                "original_filename": filename,
                "file_size": file_size,
                "file_type": file_type,
                "upload_timestamp": str(Path(file_path).stat().st_mtime),
                "file_hash": hashlib.md5(file_content).hexdigest()
            }
            
            if file_type in self.supported_types:
                try:
                    extracted_text = self.supported_types[file_type](file_path)
                    metadata["processing_successful"] = True
                except Exception as e:
                    metadata["processing_error"] = str(e)
                    metadata["processing_successful"] = False
            else:
                metadata["processing_successful"] = False
                metadata["processing_error"] = f"Unsupported file type: {file_type}"
            
            return {
                "filename": unique_filename,
                "file_path": str(file_path),
                "file_type": file_type,
                "file_size": file_size,
                "extracted_text": extracted_text,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "filename": filename,
                "file_type": "unknown",
                "file_size": 0,
                "extracted_text": "",
                "metadata": {"processing_error": str(e)}
            }
    
    def _process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            # Fallback to basic text extraction
            return "PDF file uploaded - text extraction not available (PyPDF2 not installed)"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def _process_image(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            return "Image file uploaded - OCR not available (pytesseract not installed)"
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def _process_text(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    
    def _process_doc(self, file_path: Path) -> str:
        """Extract text from DOC file"""
        try:
            import docx2txt
            text = docx2txt.process(str(file_path))
            return text.strip()
        except ImportError:
            return "DOC file uploaded - text extraction not available (python-docx not installed)"
        except Exception as e:
            return f"Error processing DOC file: {str(e)}"
    
    def _process_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            return "DOCX file uploaded - text extraction not available (python-docx not installed)"
        except Exception as e:
            return f"Error processing DOCX file: {str(e)}"
    
    def extract_flight_info(self, text: str) -> Dict[str, Any]:
        """Extract flight-related information from text"""
        import re
        
        flight_info = {
            "flight_numbers": [],
            "airlines": [],
            "dates": [],
            "airports": [],
            "delay_info": []
        }
        
        # Extract flight numbers (e.g., AC123, LH456, UA789)
        flight_pattern = r'\b([A-Z]{2,3}\s?\d{3,4})\b'
        flight_info["flight_numbers"] = re.findall(flight_pattern, text.upper())
        
        # Extract common airline codes
        airline_pattern = r'\b(Air Canada|WestJet|Lufthansa|United|American|Delta|Air France|British Airways|KLM|Iberia)\b'
        flight_info["airlines"] = re.findall(airline_pattern, text, re.IGNORECASE)
        
        # Extract dates (various formats)
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            flight_info["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract airport codes (3-letter codes)
        airport_pattern = r'\b([A-Z]{3})\b'
        potential_airports = re.findall(airport_pattern, text)
        # Filter out common non-airport 3-letter codes
        common_codes = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'NOT', 'WHAT', 'ALL', 'WERE', 'WHEN', 'YOUR', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'IF', 'UP', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME', 'HER', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'HIM', 'HAS', 'MORE', 'GO', 'NO', 'WAY', 'COULD', 'MY', 'THAN', 'FIRST', 'BEEN', 'CALL', 'WHO', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID', 'GET', 'COME', 'MADE', 'MAY', 'PART'}
        flight_info["airports"] = [code for code in potential_airports if code not in common_codes]
        
        # Extract delay information
        delay_patterns = [
            r'delayed?\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)',
            r'(\d+)\s*(hours?|hrs?|minutes?|mins?)\s*delay',
            r'delay\s+of\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)'
        ]
        for pattern in delay_patterns:
            flight_info["delay_info"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        return flight_info
    
    def cleanup_file(self, file_path: str) -> bool:
        """Delete a processed file"""
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": "File not found"}
            
            stat = path.stat()
            return {
                "filename": path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "extension": path.suffix,
                "exists": True
            }
        except Exception as e:
            return {"error": str(e), "exists": False}


# Global file processor instance
_file_processor = None

def get_file_processor() -> FileProcessor:
    """Get the global file processor instance"""
    global _file_processor
    if _file_processor is None:
        _file_processor = FileProcessor()
    return _file_processor
