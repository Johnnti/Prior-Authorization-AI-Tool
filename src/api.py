"""
Backend API for the Prior Authorization AI Tool.
Provides REST endpoints for frontend integration.
"""
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import FastAPI - provide fallback if not available
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


if FASTAPI_AVAILABLE:
    from .config import config
    from .processing_service import PAProcessingService
    
    # Pydantic models for API
    class ProcessRequest(BaseModel):
        """Request model for processing a single folder."""
        folder_name: str
        use_vision: bool = True
    
    class BatchProcessRequest(BaseModel):
        """Request model for batch processing."""
        folder_names: Optional[List[str]] = None  # None means all folders
        parallel: bool = False
    
    class ConfigUpdateRequest(BaseModel):
        """Request model for updating configuration."""
        ai_provider: Optional[str] = None
        openai_api_key: Optional[str] = None
        anthropic_api_key: Optional[str] = None
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Prior Authorization AI Tool",
        description="AI-powered tool for filling Prior Authorization forms from referral packages",
        version="1.0.0",
    )
    
    # CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize processing service
    processing_service = PAProcessingService(config)
    
    # Store for background job status
    job_status = {}
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "Prior Authorization AI Tool",
            "timestamp": datetime.now().isoformat(),
        }
    
    @app.get("/api/folders")
    async def list_folders():
        """List all available patient folders."""
        folders = processing_service.get_available_folders()
        return {
            "folders": folders,
            "total": len(folders),
            "ready_count": sum(1 for f in folders if f["ready"]),
        }
    
    @app.get("/api/folders/{folder_name}")
    async def get_folder_details(folder_name: str):
        """Get details about a specific folder."""
        folders = processing_service.get_available_folders()
        folder = next((f for f in folders if f["name"] == folder_name), None)
        
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")
        
        return folder
    
    @app.post("/api/process")
    async def process_folder(request: ProcessRequest):
        """Process a single patient folder."""
        folder_path = config.input_dir / request.folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder '{request.folder_name}' not found")
        
        result = processing_service.process_patient_folder(
            folder_path,
            use_vision=request.use_vision
        )
        
        return {
            "success": result.success,
            "summary": result.get_summary(),
            "filled_fields": [
                {"name": f.name, "value": f.value, "confidence": f.confidence}
                for f in result.filled_fields
            ],
            "uncertain_fields": [
                {"name": f.name, "value": f.value, "confidence": f.confidence}
                for f in result.uncertain_fields
            ],
            "unfilled_fields": [f.name for f in result.unfilled_fields],
            "output_path": str(result.output_path) if result.output_path else None,
            "error": result.error_message,
        }
    
    @app.post("/api/process/batch")
    async def process_batch(request: BatchProcessRequest, background_tasks: BackgroundTasks):
        """Process multiple folders (optionally in background)."""
        if request.folder_names:
            # Validate folder names
            available = {f["name"] for f in processing_service.get_available_folders()}
            invalid = set(request.folder_names) - available
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid folder names: {invalid}"
                )
        
        # Process synchronously for now
        result = processing_service.process_all_folders(parallel=request.parallel)
        
        return {
            "success": True,
            "summary": result.get_summary(),
        }
    
    @app.get("/api/results/{folder_name}")
    async def get_results(folder_name: str):
        """Get processing results for a folder."""
        output_dir = config.output_dir / folder_name
        
        if not output_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No results found for '{folder_name}'"
            )
        
        files = list(output_dir.glob("*.pdf"))
        
        return {
            "folder": folder_name,
            "files": [{"name": f.name, "path": str(f)} for f in files],
        }
    
    @app.get("/api/results/{folder_name}/download/{filename}")
    async def download_result(folder_name: str, filename: str):
        """Download a result file."""
        file_path = config.output_dir / folder_name / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=filename,
        )
    
    @app.get("/api/config")
    async def get_config():
        """Get current configuration (without sensitive data)."""
        return {
            "ai_provider": config.ai.provider,
            "openai_model": config.ai.openai_model,
            "anthropic_model": config.ai.anthropic_model,
            "has_openai_key": config.ai.openai_api_key is not None,
            "has_anthropic_key": config.ai.anthropic_api_key is not None,
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
        }
    
    @app.post("/api/config")
    async def update_config(request: ConfigUpdateRequest):
        """Update configuration."""
        if request.ai_provider:
            config.ai.provider = request.ai_provider
        if request.openai_api_key:
            config.ai.openai_api_key = request.openai_api_key
        if request.anthropic_api_key:
            config.ai.anthropic_api_key = request.anthropic_api_key
        
        return {"status": "updated"}


def create_app() -> "FastAPI":
    """Factory function to create the FastAPI app."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
    
    import uvicorn
    uvicorn.run(app, host=host, port=port)
