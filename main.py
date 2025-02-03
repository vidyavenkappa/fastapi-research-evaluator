# import os
# import time
# import pdfplumber
# import google.generativeai as genai
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv  import load_dotenv


# load_dotenv()

# # Configure Gemini API

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # Initialize FastAPI
# app = FastAPI()

# # ✅ Enable CORS for Frontend Communication
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow both local & deployed frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# PROMPT = """
# I have a PDF research paper, and I would like to evaluate its likelihood of being accepted or rejected at a leading machine learning (ML) conference, such as NeurIPS, ICLR, ICML, CoNLL, or ACL. The evaluation should be based on the standard review criteria used across these conferences, ensuring a detailed and thorough assessment that addresses both the strengths and weaknesses of the paper.

# Evaluation Criteria:
# - **Originality (1-5)**: Evaluate novelty, ideas, and contributions to ML/NLP.
# - **Soundness & Correctness (1-5)**: Check methodology, validity, and reproducibility.
# - **Clarity (1-5)**: Assess readability and organization.
# - **Meaningful Comparison (1-5)**: Compare to existing literature and baselines.
# - **Impact (1-5)**: Determine future research influence.
# - **Substance (1-5)**: Evaluate depth and scope.
# - **Replicability (1-5)**: Assess reproducibility with details provided.
# - **Appropriateness (1-5)**: Ensure alignment with ML/NLP conferences.
# - **Ethical Concerns (1-5)**: Check bias, privacy, and misuse risks.
# - **Relation to Prior Work (1-5)**: Evaluate citations and comparisons.

# ### Deliverables:
# - **Score Breakdown** (1-5 for each criterion)
# - **Reason for Acceptance**
# - **Reason for Rejection**
# - **Final Recommendation** (Accept, Reject, Borderline)
# - **Reviewer Confidence (1-5)**
# - **Final Score (1-5)**

# Please provide a structured, detailed review for the uploaded research paper.
# """

# # ✅ Function to extract text from PDF
# def extract_text_from_pdf(file_path):
#     text = ""
#     try:
#         with pdfplumber.open(file_path) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() + "\n"
#         return text.strip() if text else "Error: Unable to extract text from PDF."
#     except Exception as e:
#         return f"Error: {str(e)}"

# # ✅ Function to evaluate the research paper using Gemini
# def evaluate_paper(text, gemini_key):
#     try:
#         # ✅ Configure Gemini API with user-provided key
#         genai.configure(api_key=gemini_key)

#         # ✅ Initialize Gemini Model
#         model_name = 'gemini-1.5-flash'
#         # model = genai.GenerativeModel("gemini-2.0-pro")
#         model = genai.GenerativeModel(model_name)


#         # ✅ Generate AI Response
#         response = model.generate_content(f"{PROMPT}\n\n{text[:3000]}")  # Limit input to avoid exceeding model limits

#         return response.text if hasattr(response, "text") else "Error: No response from AI."

#     except Exception as e:
#         return f"Error in processing: {str(e)}"

# # ✅ FastAPI API Endpoint to Upload & Evaluate PDF
# @app.post("/upload/")
# async def upload_paper(file: UploadFile = File(...), gemini_key: str = Form(...)):
#     try:
#         # ✅ Save file temporarily
#         file_path = f"temp/{file.filename}"
#         os.makedirs("temp", exist_ok=True)

#         with open(file_path, "wb") as buffer:
#             buffer.write(file.file.read())

#         # ✅ Extract text from PDF
#         extracted_text = extract_text_from_pdf(file_path)

#         if extracted_text.startswith("Error:"):
#             return JSONResponse(content={"error": extracted_text}, status_code=400)

#         # ✅ Evaluate the paper using the provided Gemini API Key
#         evaluation = evaluate_paper(extracted_text, gemini_key)

#         # ✅ Cleanup the temporary file
#         os.remove(file_path)

#         return JSONResponse(content={"evaluation": evaluation})

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # ✅ Run the server with: uvicorn main:app --reload
import os
import time
import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List
from contextlib import asynccontextmanager
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TEMP_DIR = "temp"
MODEL_NAME = "gemini-1.5-flash"

PROMPT = """
I have a PDF research paper, and I would like to evaluate its likelihood of being accepted or rejected at a leading machine learning (ML) conference, such as NeurIPS, ICLR, ICML, CoNLL, or ACL. The evaluation should be based on the standard review criteria used across these conferences, ensuring a detailed and thorough assessment that addresses both the strengths and weaknesses of the paper.

Evaluation Criteria:
- **Originality (1-5)**: Evaluate novelty, ideas, and contributions to ML/NLP.
- **Soundness & Correctness (1-5)**: Check methodology, validity, and reproducibility.
- **Clarity (1-5)**: Assess readability and organization.
- **Meaningful Comparison (1-5)**: Compare to existing literature and baselines.
- **Impact (1-5)**: Determine future research influence.
- **Substance (1-5)**: Evaluate depth and scope.
- **Replicability (1-5)**: Assess reproducibility with details provided.
- **Appropriateness (1-5)**: Ensure alignment with ML/NLP conferences.
- **Ethical Concerns (1-5)**: Check bias, privacy, and misuse risks.
- **Relation to Prior Work (1-5)**: Evaluate citations and comparisons.

### Deliverables:
- **Score Breakdown** (1-5 for each criterion)
- **Reason for Acceptance**
- **Reason for Rejection**
- **Final Recommendation** (Accept, Reject, Borderline)
- **Reviewer Confidence (1-5)**
- **Final Score (1-5)**

Please provide a structured, detailed review for the uploaded research paper.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Create temporary directory and initialize resources
    logger.info("Starting up FastAPI application...")
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info(f"Created temporary directory: {TEMP_DIR}")
        
        # Initialize any other resources here
        yield
        
    finally:
        # Shutdown: Clean up resources
        logger.info("Shutting down FastAPI application...")
        try:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Research Paper Evaluator API",
    description="API for evaluating research papers using Google's Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_file(file: UploadFile) -> None:
    """Validate the uploaded file."""
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
        )

def validate_api_key(api_key: str) -> None:
    """Validate the Gemini API key."""
    if not api_key or len(api_key) < 30:
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini API key"
        )

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The file might be empty or corrupted."
            )
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text from PDF: {str(e)}"
        )

def evaluate_paper(text: str, gemini_key: str) -> str:
    """Evaluate the research paper using Gemini AI."""
    try:
        # Configure Gemini with the provided API key
        genai.configure(api_key=gemini_key)
        
        # Initialize Gemini Model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Truncate text if too long (adjust limit based on model's requirements)
        truncated_text = text[:3000]
        
        # Generate evaluation
        response = model.generate_content(f"{PROMPT}\n\n{truncated_text}")
        
        if not hasattr(response, "text") or not response.text:
            raise HTTPException(
                status_code=500,
                detail="No response received from Gemini AI"
            )
            
        return response.text
        
    except Exception as e:
        logger.error(f"Error in Gemini AI processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in AI processing: {str(e)}"
        )

@app.post("/upload/")
async def upload_paper(
    file: UploadFile = File(...),
    gemini_key: str = Form(...)
) -> JSONResponse:
    """
    Upload and evaluate a research paper.
    
    Args:
        file: The research paper file (PDF, DOC, DOCX, or TXT)
        gemini_key: Gemini AI API key
        
    Returns:
        JSONResponse containing the evaluation or error message
    """
    try:
        # Validate inputs
        validate_file(file)
        validate_api_key(gemini_key)
        
        # Create a unique filename
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        temp_filename = f"{timestamp}{file_extension}"
        file_path = os.path.join(TEMP_DIR, temp_filename)
        
        try:
            # Save file temporarily
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract and evaluate text
            extracted_text = extract_text_from_pdf(file_path)
            evaluation = evaluate_paper(extracted_text, gemini_key)
            
            return JSONResponse(
                content={"evaluation": evaluation},
                status_code=200
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except HTTPException as e:
        return JSONResponse(
            content={"error": e.detail},
            status_code=e.status_code
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            content={"error": "An unexpected error occurred"},
            status_code=500
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}