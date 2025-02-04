import os
import time
import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
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

# A default prompt (if needed), though the client-supplied prompt will be used.
DEFAULT_PROMPT = """
I have uploaded a research paper and I would like a **thorough, structured evaluation** to determine its likelihood of being accepted at a **leading ML/NLP conference**. The evaluation should be based on standard peer-review criteria. Please provide detailed feedback with scores and actionable suggestions.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    logger.info("Starting up FastAPI application...")
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info(f"Created temporary directory: {TEMP_DIR}")
        yield
    finally:
        logger.info("Shutting down FastAPI application...")
        try:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

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
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
        )

def validate_api_key(api_key: str) -> None:
    """Validate the Gemini API key."""
    if not api_key or len(api_key.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini API key"
        )

def validate_prompt(prompt: str) -> None:
    """Validate the provided prompt."""
    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

def extract_text(file_path: str, file_ext: str) -> str:
    """Extract text from the file based on its extension."""
    try:
        text = ""
        if file_ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            with open(file_path, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the file. The file might be empty or corrupted."
            )
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text: {str(e)}"
        )

def evaluate_paper(text: str, gemini_key: str, prompt: str, conference: str) -> str:
    """
    Evaluate the research paper using Gemini AI.
    The evaluation prompt is augmented with the selected conference.
    """
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(MODEL_NAME)
        # Truncate text if too long (adjust limit based on model's requirements)
        truncated_text = text[:3000]
        # Incorporate the selected conference into the prompt
        combined_prompt = (
            f"Evaluation for conference: {conference}\n\n"
            f"{prompt}\n\n"
            f"{truncated_text}"
        )
        response = model.generate_content(combined_prompt)
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
    gemini_key: str = Form(...),
    prompt: str = Form(...),
    conference: str = Form(...)
) -> JSONResponse:
    """
    Upload and evaluate a research paper.
    
    Args:
        file: The research paper file (PDF, DOC, DOCX, or TXT)
        gemini_key: Gemini AI API key
        prompt: A Markdown formatted prompt for evaluation
        conference: The conference for which the paper is being submitted
        
    Returns:
        JSONResponse containing the evaluation or error message
    """
    try:
        # Validate inputs
        validate_file(file)
        validate_api_key(gemini_key)
        validate_prompt(prompt)
        
        # Create a unique filename
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"{timestamp}{file_extension}"
        file_path = os.path.join(TEMP_DIR, temp_filename)
        
        try:
            # Save file temporarily
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text from the file based on its extension
            extracted_text = extract_text(file_path, file_extension)
            # Evaluate the paper using the provided prompt, API key, and conference
            evaluation = evaluate_paper(extracted_text, gemini_key, prompt, conference)
            
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

# To run the app:
# uvicorn main:app --reload
