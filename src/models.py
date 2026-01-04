"""
Data models for the Prior Authorization AI Tool.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pathlib import Path


class FieldStatus(Enum):
    """Status of a form field after processing."""
    FILLED = "filled"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


@dataclass
class FormField:
    """Represents a single field in the PA form."""
    name: str
    value: Optional[str] = None
    status: FieldStatus = FieldStatus.NOT_FOUND
    confidence: float = 0.0
    source_text: Optional[str] = None  # The source text this was extracted from
    page_number: Optional[int] = None


@dataclass
class ExtractedDocument:
    """Represents extracted content from a PDF document."""
    file_path: Path
    raw_text: str
    pages: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # For RAG
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PAFormTemplate:
    """Template defining expected fields in a PA form."""
    name: str
    fields: List[str]
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def get_standard_fields(cls) -> 'PAFormTemplate':
        """Returns standard PA form fields commonly found."""
        return cls(
            name="Standard PA Form",
            fields=[
                # Patient Information
                "patient_name",
                "patient_dob",
                "patient_gender",
                "patient_address",
                "patient_phone",
                "patient_id",
                "member_id",
                "group_number",
                "insurance_id",
                
                # Provider Information
                "provider_name",
                "provider_npi",
                "provider_phone",
                "provider_fax",
                "provider_address",
                "facility_name",
                "facility_npi",
                
                # Clinical Information
                "diagnosis",
                "diagnosis_code",
                "icd_10_codes",
                "procedure_code",
                "cpt_codes",
                "procedure_description",
                "medical_necessity",
                "clinical_rationale",
                
                # Medication (if applicable)
                "medication_name",
                "medication_dose",
                "medication_frequency",
                "medication_duration",
                "quantity_requested",
                
                # Service Information
                "service_type",
                "service_date",
                "service_location",
                "units_requested",
                "length_of_stay",
                
                # Additional Information
                "referring_provider",
                "ordering_provider",
                "admission_date",
                "discharge_date",
                "urgency_level",
                "previous_treatments",
            ],
            field_descriptions={
                "patient_name": "Full name of the patient",
                "patient_dob": "Patient's date of birth",
                "patient_gender": "Patient's gender",
                "member_id": "Insurance member ID number",
                "provider_npi": "National Provider Identifier",
                "diagnosis": "Primary diagnosis or condition",
                "diagnosis_code": "ICD-10 diagnosis code",
                "procedure_code": "CPT procedure code",
                "medical_necessity": "Explanation of why this treatment is medically necessary",
            }
        )


@dataclass
class ProcessingResult:
    """Result of processing a single patient's PA request."""
    patient_folder: str
    pa_form_path: Path
    referral_package_path: Path
    output_path: Optional[Path] = None
    
    # Processing results
    filled_fields: List[FormField] = field(default_factory=list)
    unfilled_fields: List[FormField] = field(default_factory=list)
    uncertain_fields: List[FormField] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the processing result."""
        total_fields = len(self.filled_fields) + len(self.unfilled_fields) + len(self.uncertain_fields)
        return {
            "patient_folder": self.patient_folder,
            "success": self.success,
            "total_fields": total_fields,
            "filled_count": len(self.filled_fields),
            "unfilled_count": len(self.unfilled_fields),
            "uncertain_count": len(self.uncertain_fields),
            "completion_rate": len(self.filled_fields) / total_fields if total_fields > 0 else 0,
            "processing_time": self.processing_time,
            "output_path": str(self.output_path) if self.output_path else None,
            "error_message": self.error_message,
        }
    
    def get_unfilled_field_names(self) -> List[str]:
        """Returns list of field names that couldn't be filled."""
        return [f.name for f in self.unfilled_fields]


@dataclass 
class BatchProcessingResult:
    """Result of processing multiple patient folders."""
    results: List[ProcessingResult] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all processing results."""
        successful = [r for r in self.results if r.success]
        return {
            "total_processed": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "total_time": self.total_time,
            "individual_results": [r.get_summary() for r in self.results],
        }
