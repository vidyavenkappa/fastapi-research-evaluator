# import os
# import time
# import pdfplumber
# import google.generativeai as genai
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from typing import List
# from contextlib import asynccontextmanager
# import logging
# import shutil

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Constants
# ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# TEMP_DIR = "temp"
# MODEL_NAME = "gemini-1.5-flash"

# PROMPT = """
# I have uploaded a research paper in PDF format, and I would like a **thorough, structured evaluation** to determine its likelihood of being accepted at a **leading ML/NLP conference** (NeurIPS, ICLR, ICML, CoNLL, ACL). The evaluation should be based on the **standard peer-review criteria** used by these conferences.  

# The review must be **specific and detailed**, referencing **actual parts of the paper** (e.g., equations, figures, tables, methodology sections). In addition to **scoring each category (1-5),** provide **clear reasons** for the score and suggest **specific improvements** to increase acceptance chances.

# ---

# ### **📌 Evaluation Criteria**
# For each category below, **assign a score from 1-5** and provide:
# - **Detailed Justification** → Why was this score assigned?  
# - **Specific Evidence** → Cite parts of the paper that support this evaluation.  
# - **Actionable Suggestions** → Provide concrete recommendations for improvement.  

# #### **1️⃣ Originality (1-5)**
# - Does the paper introduce a **novel idea, model, or approach**?  
# - How does it differ from previous work?  
# - Are there **clear innovations** or is it an **incremental improvement**?  
# - **Reference:** Identify where novelty is discussed in the paper (e.g., Section 3.1, Figure 4).  

# 🔹 **Improvement Suggestion**:  
# - If originality is weak, suggest **how the method can be differentiated** from prior work (e.g., proposing a new loss function, exploring an underrepresented dataset).  
# - Recommend **new baselines** to compare against if originality is limited.  

# ---

# #### **2️⃣ Soundness & Correctness (1-5)**
# - Is the methodology logically sound and **theoretically justified**?  
# - Are **assumptions valid**, and are there any **mathematical flaws**?  
# - Are experiments **statistically significant**, or are conclusions based on weak evidence?  

# 🔹 **Improvement Suggestion**:  
# - If there are missing proofs or weak justifications, suggest adding **mathematical derivations or additional experimental validation**.  
# - If hyperparameter tuning is absent, recommend running **additional ablation studies**.

# ---

# #### **3️⃣ Clarity (1-5)**
# - Is the paper **well-organized and easy to follow**?  
# - Are technical terms, concepts, and figures **clearly explained**?  
# - Are important **equations, tables, and graphs labeled correctly**?  

# 🔹 **Improvement Suggestion**:  
# - If unclear, suggest **rewriting sections in simpler language** or **reorganizing content for better flow**.  
# - If **notation is inconsistent**, recommend standardizing mathematical symbols.  

# ---

# #### **4️⃣ Meaningful Comparison (1-5)**
# - Does the paper **compare results with prior work**?  
# - Are comparisons **fair**, using **strong baselines**?  
# - Does it cite the most relevant papers in the field?  

# 🔹 **Improvement Suggestion**:  
# - If missing key comparisons, **suggest additional benchmarks** (e.g., add ResNet-50 if missing in a vision paper).  
# - Recommend **evaluating against newer state-of-the-art models** if only older baselines are used.  

# ---

# #### **5️⃣ Impact (1-5)**
# - How significant is the contribution?  
# - Does the paper introduce a method that can lead to **new research directions**?  

# 🔹 **Improvement Suggestion**:  
# - If impact is limited, suggest **applying the method to real-world applications** or showing **generalization to different domains**.  

# ---

# #### **6️⃣ Substance (1-5)**
# - Is the **work sufficiently detailed** to be considered substantial?  
# - Does it explore **multiple aspects of the problem**?  

# 🔹 **Improvement Suggestion**:  
# - If the paper lacks depth, recommend **adding experiments on multiple datasets** or **exploring more ablation studies**.  

# ---

# #### **7️⃣ Replicability (1-5)**
# - Can other researchers **reproduce the results** based on the provided details?  
# - Are **code, dataset, and hyperparameters included**?  

# 🔹 **Improvement Suggestion**:  
# - If replicability is poor, suggest **sharing code in a GitHub repository** and **adding dataset preprocessing details**.  

# ---

# #### **8️⃣ Appropriateness (1-5)**
# - Does the paper **align with the scope** of ML/NLP conferences?  

# 🔹 **Improvement Suggestion**:  
# - If misaligned, recommend **submitting to a more appropriate venue** (e.g., EMNLP instead of NeurIPS).  

# ---

# #### **9️⃣ Ethical Concerns (1-5)**
# - Does the paper consider **bias, fairness, or ethical risks**?  

# 🔹 **Improvement Suggestion**:  
# - If missing, recommend **analyzing biases in datasets** and **including ethical discussions**.  

# ---

# #### **🔟 Relation to Prior Work (1-5)**
# - Does the paper properly **cite and position itself** within existing research?  

# 🔹 **Improvement Suggestion**:  
# - If citations are missing, suggest adding **key references from the past 2-3 years**.  

# ---

# ### **📌 Final Recommendations**
# After scoring each category, provide:  
# ✅ **Overall Score (1-5)** → Justify why this score was assigned.  
# ✅ **Reviewer Confidence (1-5)** → Rate how confident you are in this evaluation.  
# ✅ **Reasons for Acceptance** → Highlight the strongest contributions.  
# ✅ **Reasons for Rejection** → Identify critical weaknesses.  
# ✅ **How to Improve for Acceptance** → Provide clear suggestions for a revision.  

# ---

# ### **📌 Example Review Output**
# **Originality: 3/5**  
# ✅ **Strength:** The paper proposes a transformer-based approach for multilingual text classification (**Section 3.2, Figure 5**).  
# ❌ **Weakness:** However, it does not introduce a fundamentally new concept; it mostly builds on **BERT-based models** (**Section 2.1, Related Work**).  
# 🔹 **Improvement Suggestion:**  
# - Consider extending the model to **low-resource languages** to increase novelty.  
# - Compare against **XLM-R and T5 models**, which are more competitive benchmarks.  

# **Final Score: 3.5/5 (Borderline Accept)**  
# 🔹 **Suggested Improvements for Acceptance:**  
# 1. Add a **new experimental baseline (T5)** to strengthen comparisons.  
# 2. Improve **clarity in Section 3** by explaining the dataset more clearly.  
# 3. Provide **code and hyperparameter settings** for reproducibility. 


# ### Deliverables:
# - **Score Breakdown** (1-5 for each criterion)
# - **Reason for Acceptance**
# - **Reason for Rejection**
# - **Final Recommendation** (Accept, Reject, Borderline)
# - **Reviewer Confidence (1-5)**
# - **Final Score (1-5)**

# Please provide a structured, detailed review for the uploaded research paper.
# """

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Lifespan context manager for FastAPI application.
#     Handles startup and shutdown events.
#     """
#     # Startup: Create temporary directory and initialize resources
#     logger.info("Starting up FastAPI application...")
#     try:
#         os.makedirs(TEMP_DIR, exist_ok=True)
#         logger.info(f"Created temporary directory: {TEMP_DIR}")
        
#         # Initialize any other resources here
#         yield
        
#     finally:
#         # Shutdown: Clean up resources
#         logger.info("Shutting down FastAPI application...")
#         try:
#             if os.path.exists(TEMP_DIR):
#                 shutil.rmtree(TEMP_DIR)
#                 logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")
#         except Exception as e:
#             logger.error(f"Error during cleanup: {str(e)}")

# # Initialize FastAPI with lifespan
# app = FastAPI(
#     title="Research Paper Evaluator API",
#     description="API for evaluating research papers using Google's Gemini AI",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def validate_file(file: UploadFile) -> None:
#     """Validate the uploaded file."""
#     # Check file extension
#     file_ext = os.path.splitext(file.filename)[1].lower()
#     if file_ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
#         )

#     # Check file size
#     file.file.seek(0, 2)
#     size = file.file.tell()
#     file.file.seek(0)
    
#     if size > MAX_FILE_SIZE:
#         raise HTTPException(
#             status_code=400,
#             detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
#         )

# def validate_api_key(api_key: str) -> None:
#     """Validate the Gemini API key."""
#     if not api_key or len(api_key) < 30:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid Gemini API key"
#         )

# def extract_text_from_pdf(file_path: str) -> str:
#     """Extract text from PDF file."""
#     try:
#         text = ""
#         with pdfplumber.open(file_path) as pdf:
#             for page in pdf.pages:
#                 extracted = page.extract_text()
#                 if extracted:
#                     text += extracted + "\n"
        
#         if not text.strip():
#             raise HTTPException(
#                 status_code=400,
#                 detail="Could not extract text from PDF. The file might be empty or corrupted."
#             )
            
#         return text.strip()
#     except Exception as e:
#         logger.error(f"Error extracting text from PDF: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error extracting text from PDF: {str(e)}"
#         )

# def evaluate_paper(text: str, gemini_key: str) -> str:
#     """Evaluate the research paper using Gemini AI."""
#     try:
#         # Configure Gemini with the provided API key
#         genai.configure(api_key=gemini_key)
        
#         # Initialize Gemini Model
#         model = genai.GenerativeModel(MODEL_NAME)
        
#         # Truncate text if too long (adjust limit based on model's requirements)
#         truncated_text = text[:3000]
        
#         # Generate evaluation
#         response = model.generate_content(f"{PROMPT}\n\n{truncated_text}")
        
#         if not hasattr(response, "text") or not response.text:
#             raise HTTPException(
#                 status_code=500,
#                 detail="No response received from Gemini AI"
#             )
            
#         return response.text
        
#     except Exception as e:
#         logger.error(f"Error in Gemini AI processing: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error in AI processing: {str(e)}"
#         )

# @app.post("/upload/")
# async def upload_paper(
#     file: UploadFile = File(...),
#     gemini_key: str = Form(...)
# ) -> JSONResponse:
#     """
#     Upload and evaluate a research paper.
    
#     Args:
#         file: The research paper file (PDF, DOC, DOCX, or TXT)
#         gemini_key: Gemini AI API key
        
#     Returns:
#         JSONResponse containing the evaluation or error message
#     """
#     try:
#         # Validate inputs
#         validate_file(file)
#         validate_api_key(gemini_key)
        
#         # Create a unique filename
#         timestamp = int(time.time())
#         file_extension = os.path.splitext(file.filename)[1]
#         temp_filename = f"{timestamp}{file_extension}"
#         file_path = os.path.join(TEMP_DIR, temp_filename)
        
#         try:
#             # Save file temporarily
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
            
#             # Extract and evaluate text
#             extracted_text = extract_text_from_pdf(file_path)
#             evaluation = evaluate_paper(extracted_text, gemini_key)
            
#             return JSONResponse(
#                 content={"evaluation": evaluation},
#                 status_code=200
#             )
            
#         finally:
#             # Clean up temporary file
#             if os.path.exists(file_path):
#                 os.remove(file_path)
                
#     except HTTPException as e:
#         return JSONResponse(
#             content={"error": e.detail},
#             status_code=e.status_code
#         )
        
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return JSONResponse(
#             content={"error": "An unexpected error occurred"},
#             status_code=500
#         )

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {"status": "healthy"}
# #uvicorn main:app --reload

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
