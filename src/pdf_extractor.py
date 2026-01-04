"""
PDF extraction utilities for extracting text and images from PDF documents.
Supports both text-based PDFs and scanned documents via OCR.
"""
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts text and images from PDF documents.
    Uses multiple extraction strategies for maximum compatibility.
    """
    
    def __init__(self, use_ocr: bool = True, dpi: int = 200):
        self.use_ocr = use_ocr
        self.dpi = dpi
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and import required dependencies."""
        self.pdfplumber = None
        self.fitz = None  # PyMuPDF
        self.PIL = None
        
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")
        
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
        
        try:
            from PIL import Image
            self.PIL = Image
        except ImportError:
            logger.warning("Pillow not installed. Install with: pip install pillow")
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract all text from a PDF document.
        Tries multiple methods for best results.
        """
        text = ""
        
        # Try pdfplumber first (good for text-based PDFs)
        if self.pdfplumber:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Try PyMuPDF
        if self.fitz:
            try:
                text = self._extract_with_pymupdf(pdf_path)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        return text
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        with self.pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF."""
        text_parts = []
        doc = self.fitz.open(pdf_path)
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    
    def extract_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract content from each page of a PDF.
        Returns list of page dictionaries with text and metadata.
        """
        pages = []
        
        if self.pdfplumber:
            try:
                with self.pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_data = {
                            "page_number": i + 1,
                            "text": page.extract_text() or "",
                            "width": page.width,
                            "height": page.height,
                            "tables": self._extract_tables(page),
                        }
                        pages.append(page_data)
            except Exception as e:
                logger.error(f"Error extracting pages: {e}")
        
        return pages
    
    def _extract_tables(self, page) -> List[List[List[str]]]:
        """Extract tables from a pdfplumber page."""
        tables = []
        try:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                tables = extracted_tables
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        return tables
    
    def convert_to_images(self, pdf_path: Path) -> List[bytes]:
        """
        Convert PDF pages to images for OCR or vision processing.
        Returns list of image bytes (PNG format).
        """
        images = []
        
        if self.fitz and self.PIL:
            try:
                doc = self.fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Render at specified DPI
                    mat = self.fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image then to bytes
                    img = self.PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="PNG")
                    images.append(img_buffer.getvalue())
                
                doc.close()
            except Exception as e:
                logger.error(f"Error converting PDF to images: {e}")
        
        return images
    
    def get_images_as_base64(self, pdf_path: Path) -> List[str]:
        """
        Convert PDF pages to base64-encoded images.
        Useful for sending to vision AI models.
        """
        images = self.convert_to_images(pdf_path)
        return [base64.b64encode(img).decode('utf-8') for img in images]
    
    def get_form_fields(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract fillable form fields from a PDF.
        Returns list of field dictionaries.
        """
        fields = []
        
        if self.fitz:
            try:
                doc = self.fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    widgets = page.widgets()
                    if widgets:
                        for widget in widgets:
                            field_info = {
                                "page": page_num + 1,
                                "field_name": widget.field_name,
                                "field_type": widget.field_type_string,
                                "field_value": widget.field_value,
                                "rect": list(widget.rect),
                            }
                            fields.append(field_info)
                doc.close()
            except Exception as e:
                logger.warning(f"Error extracting form fields: {e}")
        
        return fields


class TextChunker:
    """
    Chunks text for RAG-style retrieval.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Returns list of chunk dictionaries with text and metadata.
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end),
                    )
                    if sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "metadata": metadata or {},
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk content from multiple pages.
        Preserves page number information in metadata.
        """
        all_chunks = []
        
        for page in pages:
            page_metadata = {
                "page_number": page.get("page_number", 0),
                "source": "page_content",
            }
            page_chunks = self.chunk_text(page.get("text", ""), page_metadata)
            all_chunks.extend(page_chunks)
        
        # Re-number chunks
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = i
        
        return all_chunks
