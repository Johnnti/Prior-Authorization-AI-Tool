"""
Main processing service that orchestrates the PA form filling workflow.
"""
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import AppConfig, config as default_config
from .models import (
    ProcessingResult,
    BatchProcessingResult,
    PAFormTemplate,
    ExtractedDocument,
    FormField,
    FieldStatus,
)
from .pdf_extractor import PDFExtractor, TextChunker
from .ai_extractor import AIExtractor, RAGRetriever
from .pdf_filler import PDFFormFiller

logger = logging.getLogger(__name__)


class PAProcessingService:
    """
    Main service for processing Prior Authorization requests.
    Orchestrates PDF extraction, AI analysis, and form filling.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or default_config
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(
            use_ocr=self.config.processing.use_ocr,
            dpi=self.config.processing.pdf_dpi
        )
        self.text_chunker = TextChunker(
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap
        )
        self.ai_extractor = None  # Lazy initialization
        self.pdf_filler = PDFFormFiller()
        self.rag_retriever = RAGRetriever()
        
        # Form template
        self.form_template = PAFormTemplate.get_standard_fields()
    
    def _ensure_ai_extractor(self):
        """Initialize AI extractor if not already done."""
        if self.ai_extractor is None:
            self.ai_extractor = AIExtractor(self.config.ai)
    
    def process_patient_folder(
        self,
        folder_path: Path,
        use_vision: bool = True
    ) -> ProcessingResult:
        """
        Process a single patient's folder containing PA form and referral package.
        
        Args:
            folder_path: Path to the patient's folder
            use_vision: Whether to use vision AI for image analysis
        
        Returns:
            ProcessingResult with all extracted and filled information
        """
        start_time = time.time()
        folder_name = folder_path.name
        
        logger.info(f"Processing patient folder: {folder_name}")
        
        # Find the PA form and referral package
        pa_form_path = self._find_pa_form(folder_path)
        referral_path = self._find_referral_package(folder_path)
        
        result = ProcessingResult(
            patient_folder=folder_name,
            pa_form_path=pa_form_path,
            referral_package_path=referral_path,
        )
        
        if not pa_form_path or not referral_path:
            result.error_message = f"Missing files: PA form={pa_form_path is not None}, Referral={referral_path is not None}"
            result.processing_time = time.time() - start_time
            return result
        
        try:
            # Step 1: Extract text from referral package
            logger.info(f"Extracting text from referral package...")
            referral_text = self.pdf_extractor.extract_text(referral_path)
            referral_pages = self.pdf_extractor.extract_pages(referral_path)
            
            # Step 2: Create chunks for RAG
            chunks = self.text_chunker.chunk_pages(referral_pages)
            self.rag_retriever.index_chunks(chunks)
            
            # Step 3: Extract information using AI
            logger.info(f"Extracting information using AI...")
            self._ensure_ai_extractor()
            
            if use_vision and referral_text.strip() == "":
                # Scanned document - use vision
                logger.info("Using vision AI for scanned document...")
                images_b64 = self.pdf_extractor.get_images_as_base64(referral_path)
                extracted_fields = self.ai_extractor.extract_from_images(
                    images_b64,
                    self.form_template.fields,
                    self.form_template.field_descriptions,
                )
            else:
                # Text-based document
                # Use RAG to get relevant chunks for each field
                field_chunks = self.rag_retriever.retrieve_for_fields(
                    self.form_template.fields,
                    self.form_template.field_descriptions,
                    top_k=3
                )
                
                # Combine relevant context
                context_parts = [referral_text[:5000]]  # Include beginning of document
                seen_chunks = set()
                for field, chunks in field_chunks.items():
                    for chunk in chunks:
                        chunk_id = chunk["chunk_id"]
                        if chunk_id not in seen_chunks:
                            context_parts.append(chunk["text"])
                            seen_chunks.add(chunk_id)
                
                combined_context = "\n\n---\n\n".join(context_parts)
                
                extracted_fields = self.ai_extractor.extract_from_text(
                    combined_context,
                    self.form_template.fields,
                    self.form_template.field_descriptions,
                )
            
            # Step 4: Categorize fields
            for field in extracted_fields:
                if field.status == FieldStatus.FILLED:
                    result.filled_fields.append(field)
                elif field.status == FieldStatus.UNCERTAIN:
                    result.uncertain_fields.append(field)
                else:
                    result.unfilled_fields.append(field)
            
            # Step 5: Fill the PA form
            logger.info(f"Filling PA form...")
            output_filename = f"filled_PA_{folder_name}.pdf"
            output_path = self.config.output_dir / folder_name / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to fill the original form
            fill_success = self.pdf_filler.fill_form(
                pa_form_path,
                output_path,
                extracted_fields,
            )
            
            # Also create a detailed report
            report_path = self.config.output_dir / folder_name / f"extraction_report_{folder_name}.pdf"
            self.pdf_filler.create_filled_report(
                report_path,
                extracted_fields,
                {"folder": folder_name}
            )
            
            result.output_path = output_path
            result.success = True
            
            logger.info(f"Successfully processed {folder_name}: "
                       f"{len(result.filled_fields)} filled, "
                       f"{len(result.uncertain_fields)} uncertain, "
                       f"{len(result.unfilled_fields)} not found")
            
        except Exception as e:
            logger.error(f"Error processing {folder_name}: {e}")
            result.error_message = str(e)
        
        result.processing_time = time.time() - start_time
        return result
    
    def _find_pa_form(self, folder_path: Path) -> Optional[Path]:
        """Find the PA form in a folder (case-insensitive)."""
        for pattern in ["PA.pdf", "pa.pdf", "PA*.pdf", "pa*.pdf"]:
            matches = list(folder_path.glob(pattern))
            if matches:
                return matches[0]
        return None
    
    def _find_referral_package(self, folder_path: Path) -> Optional[Path]:
        """Find the referral package in a folder."""
        for pattern in ["referral_package.pdf", "Referral_Package.pdf", "referral*.pdf"]:
            matches = list(folder_path.glob(pattern))
            if matches:
                return matches[0]
        return None
    
    def process_all_folders(
        self,
        input_dir: Optional[Path] = None,
        parallel: bool = False,
        max_workers: int = 3
    ) -> BatchProcessingResult:
        """
        Process all patient folders in the input directory.
        
        Args:
            input_dir: Directory containing patient folders (default: config.input_dir)
            parallel: Whether to process folders in parallel
            max_workers: Maximum number of parallel workers
        
        Returns:
            BatchProcessingResult with all results
        """
        start_time = time.time()
        input_dir = input_dir or self.config.input_dir
        
        # Get all patient folders
        folders = [f for f in input_dir.iterdir() if f.is_dir()]
        
        if not folders:
            logger.warning(f"No patient folders found in {input_dir}")
            return BatchProcessingResult()
        
        logger.info(f"Found {len(folders)} patient folders to process")
        
        results = []
        
        if parallel and len(folders) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_patient_folder, folder): folder
                    for folder in folders
                }
                
                for future in as_completed(futures):
                    folder = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {folder.name}: {e}")
                        results.append(ProcessingResult(
                            patient_folder=folder.name,
                            pa_form_path=folder,
                            referral_package_path=folder,
                            error_message=str(e)
                        ))
        else:
            # Sequential processing
            for folder in folders:
                result = self.process_patient_folder(folder)
                results.append(result)
        
        batch_result = BatchProcessingResult(
            results=results,
            total_time=time.time() - start_time,
        )
        
        # Log summary
        summary = batch_result.get_summary()
        logger.info(f"Batch processing complete: "
                   f"{summary['successful']}/{summary['total_processed']} successful, "
                   f"total time: {summary['total_time']:.2f}s")
        
        return batch_result
    
    def get_available_folders(self, input_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Get list of available patient folders with their file status.
        Useful for frontend to display available patients.
        """
        input_dir = input_dir or self.config.input_dir
        
        folders = []
        for folder in input_dir.iterdir():
            if folder.is_dir():
                pa_form = self._find_pa_form(folder)
                referral = self._find_referral_package(folder)
                
                folders.append({
                    "name": folder.name,
                    "path": str(folder),
                    "has_pa_form": pa_form is not None,
                    "has_referral_package": referral is not None,
                    "ready": pa_form is not None and referral is not None,
                })
        
        return folders
