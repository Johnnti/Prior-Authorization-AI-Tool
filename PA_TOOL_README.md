# Prior Authorization AI Tool

An AI-powered tool for automatically filling Prior Authorization (PA) forms by extracting information from referral packages. This tool uses advanced AI models with OCR and vision capabilities to intelligently extract patient information, clinical details, and other relevant data from medical documents.

## Features

- 📄 **PDF Extraction**: Extract text and tables from both text-based and scanned PDF documents
- 🤖 **AI-Powered Analysis**: Uses GPT-4o or Claude with vision capabilities for intelligent data extraction
- 🔍 **RAG-Style Retrieval**: Smart chunking and retrieval for accurate field matching
- 📝 **Form Filling**: Automatically fills PDF forms or creates detailed extraction reports
- 🖥️ **Backend API**: RESTful API ready for frontend integration
- 📊 **Detailed Reporting**: Shows filled fields, uncertain values, and missing information

## Project Structure

```
Prior-Authorization-AI-Tool/
├── main.py                 # Main entry point / CLI
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models
│   ├── pdf_extractor.py   # PDF text/image extraction
│   ├── ai_extractor.py    # AI-powered data extraction
│   ├── pdf_filler.py      # PDF form filling
│   ├── processing_service.py  # Main processing orchestration
│   └── api.py             # REST API endpoints
├── Input Data/            # Patient folders with PA forms and referral packages
│   ├── Adbulla/
│   │   ├── PA.pdf
│   │   └── referral_package.pdf
│   ├── Akshay/
│   └── Amy/
└── Output/                # Generated filled forms and reports
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Prior-Authorization-AI-Tool
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key(s)
   ```

## Configuration

### API Keys

Set your AI provider API key as an environment variable or pass it via command line:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here

# Option 2: Command line argument
python main.py --all --openai-key your_key_here
```

### Supported AI Providers

- **OpenAI**: GPT-4o (recommended for best vision capabilities)
- **Anthropic**: Claude claude-sonnet-4-20250514 (excellent for document understanding)

## Usage

### Command Line Interface

```bash
# List available patient folders
python main.py --list

# Process a single patient folder
python main.py --folder Adbulla

# Process all patient folders
python main.py --all

# Process with parallel execution
python main.py --all --parallel

# Use specific AI provider
python main.py --all --provider anthropic --anthropic-key YOUR_KEY

# Disable vision AI (use only text extraction)
python main.py --folder Adbulla --no-vision
```

### API Server

Start the API server for frontend integration:

```bash
# Start server on default port (8000)
python main.py --server

# Start on custom host/port
python main.py --server --host 127.0.0.1 --port 3000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/folders` | GET | List available patient folders |
| `/api/folders/{name}` | GET | Get folder details |
| `/api/process` | POST | Process a single folder |
| `/api/process/batch` | POST | Process multiple folders |
| `/api/results/{name}` | GET | Get processing results |
| `/api/results/{name}/download/{file}` | GET | Download result file |
| `/api/config` | GET/POST | Get or update configuration |

### Programmatic Usage

```python
from src import PAProcessingService, config

# Initialize service
service = PAProcessingService(config)

# List available folders
folders = service.get_available_folders()
print(folders)

# Process a single folder
from pathlib import Path
result = service.process_patient_folder(Path("Input Data/Adbulla"))

# Check results
print(f"Filled: {len(result.filled_fields)}")
print(f"Missing: {result.get_unfilled_field_names()}")

# Process all folders
batch_result = service.process_all_folders()
print(batch_result.get_summary())
```

## Input Data Format

Each patient folder should contain:
- `PA.pdf` or `pa.pdf` - The Prior Authorization form to be filled
- `referral_package.pdf` - The source document containing patient/clinical information

## Output

For each processed patient, the tool generates:
1. **Filled PA Form** (`filled_PA_{name}.pdf`) - The PA form with extracted data
2. **Extraction Report** (`extraction_report_{name}.pdf`) - Detailed report showing:
   - All successfully extracted fields with confidence scores
   - Uncertain fields requiring manual review
   - Fields not found in the referral package

## Fields Extracted

The tool attempts to extract the following information:

### Patient Information
- Name, DOB, Gender, Address, Phone
- Member ID, Group Number, Insurance ID

### Provider Information
- Provider Name, NPI, Contact Info
- Facility Name and NPI

### Clinical Information
- Diagnosis and ICD-10 Codes
- Procedure Codes (CPT)
- Medical Necessity/Clinical Rationale

### Service Details
- Service Type, Date, Location
- Medication Details (if applicable)
- Units/Duration Requested

## Frontend Integration

The API is designed for easy frontend integration:

```javascript
// Example: Process a patient folder
const response = await fetch('http://localhost:8000/api/process', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    folder_name: 'Adbulla',
    use_vision: true
  })
});

const result = await response.json();
console.log(result.filled_fields);
console.log(result.unfilled_fields);
```

## Troubleshooting

### Common Issues

1. **"No text extracted"**: The PDF may be scanned. Ensure `--vision` is enabled (default).

2. **"API key not set"**: Set your API key via environment variable or command line.

3. **"Module not found"**: Run `pip install -r requirements.txt`

4. **Low confidence scores**: The document may have poor quality. Try increasing DPI in config.

### Logging

Enable debug logging for troubleshooting:
```bash
python main.py --all --log-level DEBUG
```

## License

MIT License - See LICENSE file for details.
