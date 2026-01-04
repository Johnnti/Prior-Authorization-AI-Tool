"""
PDF Form Filler - Fills PDF forms with extracted data.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import FormField, FieldStatus

logger = logging.getLogger(__name__)


class PDFFormFiller:
    """
    Fills PDF forms with extracted data.
    Supports both fillable PDF forms and overlay-based filling.
    """
    
    def __init__(self):
        self.fitz = None
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and import required dependencies."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
    
    def fill_form(
        self,
        template_path: Path,
        output_path: Path,
        fields: List[FormField],
        field_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Fill a PDF form with extracted field values.
        
        Args:
            template_path: Path to the original PDF form
            output_path: Path for the filled output PDF
            fields: List of FormField objects with values
            field_mapping: Optional mapping from field names to PDF field names
        
        Returns:
            True if successful, False otherwise
        """
        if not self.fitz:
            logger.error("PyMuPDF not available for form filling")
            return False
        
        field_mapping = field_mapping or {}
        
        try:
            # Open the template
            doc = self.fitz.open(template_path)
            
            # Try to fill form fields first
            filled_any = self._fill_form_fields(doc, fields, field_mapping)
            
            # If no form fields found or filled, try overlay approach
            if not filled_any:
                logger.info("No fillable form fields found, using text overlay")
                self._add_text_overlay(doc, fields)
            
            # Save the filled form
            doc.save(output_path)
            doc.close()
            
            logger.info(f"Saved filled form to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error filling PDF form: {e}")
            return False
    
    def _fill_form_fields(
        self,
        doc,
        fields: List[FormField],
        field_mapping: Dict[str, str]
    ) -> bool:
        """
        Fill PDF form fields if they exist.
        Returns True if any fields were filled.
        """
        filled_any = False
        
        # Create a lookup from field values
        field_values = {f.name: f.value for f in fields if f.value and f.status == FieldStatus.FILLED}
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get all widgets (form fields) on this page
            widgets = list(page.widgets())
            
            for widget in widgets:
                pdf_field_name = widget.field_name
                
                # Try to match with our extracted fields
                matched_value = None
                
                # Direct match
                if pdf_field_name in field_values:
                    matched_value = field_values[pdf_field_name]
                
                # Try mapping
                elif pdf_field_name in field_mapping:
                    mapped_name = field_mapping[pdf_field_name]
                    if mapped_name in field_values:
                        matched_value = field_values[mapped_name]
                
                # Try fuzzy matching
                else:
                    matched_value = self._fuzzy_match_field(pdf_field_name, field_values)
                
                if matched_value:
                    try:
                        widget.field_value = matched_value
                        widget.update()
                        filled_any = True
                        logger.debug(f"Filled field '{pdf_field_name}' with '{matched_value}'")
                    except Exception as e:
                        logger.warning(f"Could not fill field '{pdf_field_name}': {e}")
        
        return filled_any
    
    def _fuzzy_match_field(self, pdf_field_name: str, field_values: Dict[str, str]) -> Optional[str]:
        """
        Try to fuzzy match a PDF field name with extracted field names.
        """
        pdf_field_lower = pdf_field_name.lower().replace(" ", "_").replace("-", "_")
        
        for field_name, value in field_values.items():
            field_lower = field_name.lower()
            
            # Check for substring matches
            if field_lower in pdf_field_lower or pdf_field_lower in field_lower:
                return value
            
            # Check for common variations
            variations = [
                ("name", "patient_name"),
                ("dob", "patient_dob"),
                ("date_of_birth", "patient_dob"),
                ("member", "member_id"),
                ("provider", "provider_name"),
                ("npi", "provider_npi"),
                ("diagnosis", "diagnosis"),
                ("icd", "icd_10_codes"),
            ]
            
            for keyword, mapped_field in variations:
                if keyword in pdf_field_lower and mapped_field == field_lower:
                    return value
        
        return None
    
    def _add_text_overlay(self, doc, fields: List[FormField]):
        """
        Add extracted data as a text overlay on the first page.
        Used when the PDF doesn't have fillable form fields.
        """
        # Get filled fields
        filled_fields = [f for f in fields if f.value and f.status in (FieldStatus.FILLED, FieldStatus.UNCERTAIN)]
        
        if not filled_fields:
            return
        
        # Add a new page with extracted information
        page = doc.new_page(-1)  # Add at end
        
        # Build the text content
        text_lines = [
            "=" * 60,
            "EXTRACTED INFORMATION FROM REFERRAL PACKAGE",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]
        
        for field in filled_fields:
            confidence_str = f" (confidence: {field.confidence:.0%})" if field.confidence < 1.0 else ""
            text_lines.append(f"{field.name}: {field.value}{confidence_str}")
        
        text_content = "\n".join(text_lines)
        
        # Insert text on the page
        rect = self.fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(
            rect,
            text_content,
            fontsize=10,
            fontname="helv",
        )
    
    def create_filled_report(
        self,
        output_path: Path,
        fields: List[FormField],
        patient_info: Dict[str, Any]
    ) -> bool:
        """
        Create a new PDF report with all extracted information.
        Useful when the original form can't be filled.
        """
        if not self.fitz:
            logger.error("PyMuPDF not available")
            return False
        
        try:
            doc = self.fitz.open()  # Create new document
            
            # Categorize fields
            filled = [f for f in fields if f.status == FieldStatus.FILLED]
            uncertain = [f for f in fields if f.status == FieldStatus.UNCERTAIN]
            not_found = [f for f in fields if f.status == FieldStatus.NOT_FOUND]
            
            # Create first page - Filled Fields
            self._add_report_page(
                doc,
                "Prior Authorization - Extracted Information",
                filled,
                patient_info
            )
            
            # Create second page - Uncertain and Missing Fields
            if uncertain or not_found:
                self._add_status_page(doc, uncertain, not_found)
            
            doc.save(output_path)
            doc.close()
            
            logger.info(f"Created filled report: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return False
    
    def _add_report_page(
        self,
        doc,
        title: str,
        fields: List[FormField],
        patient_info: Dict[str, Any]
    ):
        """Add a report page to the document."""
        page = doc.new_page()
        
        y_pos = 50
        
        # Title
        page.insert_text(
            (50, y_pos),
            title,
            fontsize=16,
            fontname="helv",
        )
        y_pos += 30
        
        # Timestamp
        page.insert_text(
            (50, y_pos),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=10,
            fontname="helv",
        )
        y_pos += 20
        
        # Patient folder
        if "folder" in patient_info:
            page.insert_text(
                (50, y_pos),
                f"Patient Folder: {patient_info['folder']}",
                fontsize=10,
                fontname="helv",
            )
            y_pos += 30
        
        # Divider
        page.draw_line((50, y_pos), (page.rect.width - 50, y_pos))
        y_pos += 20
        
        # Fields
        for field in fields:
            if y_pos > page.rect.height - 50:
                # Need new page
                page = doc.new_page()
                y_pos = 50
            
            confidence_str = f" [{field.confidence:.0%}]" if field.confidence < 1.0 else ""
            text = f"{field.name}: {field.value}{confidence_str}"
            
            page.insert_text(
                (50, y_pos),
                text,
                fontsize=10,
                fontname="helv",
            )
            y_pos += 15
    
    def _add_status_page(
        self,
        doc,
        uncertain: List[FormField],
        not_found: List[FormField]
    ):
        """Add a page showing uncertain and missing fields."""
        page = doc.new_page()
        
        y_pos = 50
        
        # Title
        page.insert_text(
            (50, y_pos),
            "Fields Requiring Review",
            fontsize=16,
            fontname="helv",
        )
        y_pos += 40
        
        # Uncertain fields
        if uncertain:
            page.insert_text(
                (50, y_pos),
                "UNCERTAIN VALUES (Low Confidence):",
                fontsize=12,
                fontname="helv",
            )
            y_pos += 20
            
            for field in uncertain:
                text = f"  • {field.name}: {field.value} [{field.confidence:.0%}]"
                page.insert_text((50, y_pos), text, fontsize=10, fontname="helv")
                y_pos += 15
            
            y_pos += 20
        
        # Not found fields
        if not_found:
            page.insert_text(
                (50, y_pos),
                "FIELDS NOT FOUND IN REFERRAL PACKAGE:",
                fontsize=12,
                fontname="helv",
            )
            y_pos += 20
            
            for field in not_found:
                text = f"  • {field.name}"
                page.insert_text((50, y_pos), text, fontsize=10, fontname="helv")
                y_pos += 15
