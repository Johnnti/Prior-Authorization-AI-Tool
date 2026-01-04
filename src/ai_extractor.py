"""
AI-powered extraction service for extracting structured data from documents.
Supports OpenAI GPT-4 Vision and Anthropic Claude models.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import FormField, FieldStatus, PAFormTemplate, ExtractedDocument
from .config import AIConfig

logger = logging.getLogger(__name__)


class AIExtractor:
    """
    AI-powered document extraction service.
    Uses vision-capable models to extract structured data from documents.
    """
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the AI client based on configuration."""
        if self.config.provider == "openai":
            self._setup_openai()
        elif self.config.provider == "anthropic":
            self._setup_anthropic()
        else:
            raise ValueError(f"Unsupported AI provider: {self.config.provider}")
    
    def _setup_openai(self):
        """Setup OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.openai_api_key)
            self.model = self.config.openai_model
            logger.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _setup_anthropic(self):
        """Setup Anthropic client."""
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.config.anthropic_api_key)
            self.model = self.config.anthropic_model
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    def extract_from_text(
        self,
        text: str,
        fields: List[str],
        field_descriptions: Optional[Dict[str, str]] = None
    ) -> List[FormField]:
        """
        Extract field values from text using AI.
        """
        field_descriptions = field_descriptions or {}
        
        # Build the extraction prompt
        prompt = self._build_extraction_prompt(text, fields, field_descriptions)
        
        # Call the AI model
        if self.config.provider == "openai":
            response = self._call_openai_text(prompt)
        else:
            response = self._call_anthropic_text(prompt)
        
        # Parse the response
        return self._parse_extraction_response(response, fields)
    
    def extract_from_images(
        self,
        images_base64: List[str],
        fields: List[str],
        field_descriptions: Optional[Dict[str, str]] = None,
        additional_context: Optional[str] = None
    ) -> List[FormField]:
        """
        Extract field values from images using vision AI.
        """
        field_descriptions = field_descriptions or {}
        
        # Build the extraction prompt
        prompt = self._build_extraction_prompt(
            additional_context or "Extract information from the following document images.",
            fields,
            field_descriptions
        )
        
        # Call the AI model with images
        if self.config.provider == "openai":
            response = self._call_openai_vision(prompt, images_base64)
        else:
            response = self._call_anthropic_vision(prompt, images_base64)
        
        # Parse the response
        return self._parse_extraction_response(response, fields)
    
    def _build_extraction_prompt(
        self,
        context: str,
        fields: List[str],
        field_descriptions: Dict[str, str]
    ) -> str:
        """Build the extraction prompt for the AI model."""
        
        fields_info = []
        for field in fields:
            desc = field_descriptions.get(field, "")
            if desc:
                fields_info.append(f"- {field}: {desc}")
            else:
                fields_info.append(f"- {field}")
        
        fields_list = "\n".join(fields_info)
        
        prompt = f"""You are an expert medical document analyst. Your task is to extract specific information from a medical referral package document.

DOCUMENT CONTENT:
{context}

FIELDS TO EXTRACT:
{fields_list}

INSTRUCTIONS:
1. Carefully analyze the document content
2. Extract values for each field listed above
3. If a field's value is not found in the document, mark it as "NOT_FOUND"
4. If you're uncertain about a value, mark confidence as low
5. Be precise - only extract information that is explicitly stated

RESPONSE FORMAT:
Return a JSON object with the following structure:
{{
    "extracted_fields": [
        {{
            "name": "field_name",
            "value": "extracted value or NOT_FOUND",
            "confidence": 0.0 to 1.0,
            "source_text": "the exact text this was extracted from (if applicable)"
        }}
    ]
}}

Return ONLY the JSON object, no additional text."""
        
        return prompt
    
    def _call_openai_text(self, prompt: str) -> str:
        """Call OpenAI API with text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical document extraction assistant. Extract information accurately and return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content
    
    def _call_openai_vision(self, prompt: str, images_base64: List[str]) -> str:
        """Call OpenAI API with images."""
        # Build content with images
        content = [{"type": "text", "text": prompt}]
        
        for img_b64 in images_base64[:10]:  # Limit to 10 images
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical document extraction assistant with vision capabilities. Extract information accurately from documents and return valid JSON."
                },
                {"role": "user", "content": content}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content
    
    def _call_anthropic_text(self, prompt: str) -> str:
        """Call Anthropic API with text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a medical document extraction assistant. Extract information accurately and return valid JSON."
        )
        return response.content[0].text
    
    def _call_anthropic_vision(self, prompt: str, images_base64: List[str]) -> str:
        """Call Anthropic API with images."""
        # Build content with images
        content = []
        
        for img_b64 in images_base64[:10]:  # Limit to 10 images
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64
                }
            })
        
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "user", "content": content}
            ],
            system="You are a medical document extraction assistant with vision capabilities. Extract information accurately from documents and return valid JSON."
        )
        return response.content[0].text
    
    def _parse_extraction_response(self, response: str, fields: List[str]) -> List[FormField]:
        """Parse the AI response into FormField objects."""
        form_fields = []
        
        try:
            # Clean the response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            data = json.loads(response)
            extracted = data.get("extracted_fields", [])
            
            # Create a lookup for extracted fields
            extracted_lookup = {f["name"]: f for f in extracted}
            
            for field_name in fields:
                if field_name in extracted_lookup:
                    field_data = extracted_lookup[field_name]
                    value = field_data.get("value")
                    confidence = field_data.get("confidence", 0.5)
                    
                    if value and value.upper() != "NOT_FOUND":
                        status = FieldStatus.FILLED if confidence >= 0.7 else FieldStatus.UNCERTAIN
                    else:
                        status = FieldStatus.NOT_FOUND
                        value = None
                    
                    form_fields.append(FormField(
                        name=field_name,
                        value=value,
                        status=status,
                        confidence=confidence,
                        source_text=field_data.get("source_text"),
                    ))
                else:
                    form_fields.append(FormField(
                        name=field_name,
                        status=FieldStatus.NOT_FOUND,
                    ))
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response was: {response}")
            # Return empty fields for all
            for field_name in fields:
                form_fields.append(FormField(
                    name=field_name,
                    status=FieldStatus.NOT_FOUND,
                ))
        
        return form_fields


class RAGRetriever:
    """
    Simple RAG-style retriever for finding relevant chunks.
    Uses keyword matching and TF-IDF-like scoring.
    """
    
    def __init__(self):
        self.chunks = []
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks for retrieval."""
        self.chunks = chunks
        # Pre-process chunks for faster retrieval
        for chunk in self.chunks:
            chunk["text_lower"] = chunk["text"].lower()
            chunk["words"] = set(chunk["text_lower"].split())
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query.
        Uses simple keyword matching.
        """
        if not self.chunks:
            return []
        
        query_words = set(query.lower().split())
        
        # Score each chunk
        scored_chunks = []
        for chunk in self.chunks:
            # Calculate overlap score
            overlap = len(query_words & chunk["words"])
            if overlap > 0:
                score = overlap / len(query_words)
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]
    
    def retrieve_for_fields(
        self,
        fields: List[str],
        field_descriptions: Dict[str, str],
        top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant chunks for each field.
        Returns a dictionary mapping field names to relevant chunks.
        """
        field_chunks = {}
        
        for field in fields:
            # Build query from field name and description
            query_parts = [field.replace("_", " ")]
            if field in field_descriptions:
                query_parts.append(field_descriptions[field])
            query = " ".join(query_parts)
            
            field_chunks[field] = self.retrieve(query, top_k)
        
        return field_chunks
