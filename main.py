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
MAX_FILE_SIZE = 20 * 1024 * 1024  # 10MB
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

def validate_review(prompt: str) -> None:
    """Validate the provided prompt."""
    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
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

def evaluate_paper(text: str, gemini_key: str, conference: str, add_prompt: str = None,) -> str:
    """
    Evaluate the research paper using Gemini AI.
    The evaluation prompt is augmented with the selected conference.
    """
    try:

        prompt = """
        
        
📌 Research Paper Evaluation Prompt (For Critical, Human-Like Review)
🔍 Goal:
Evaluate the research paper (PDF format) critically and rigorously, identifying fundamental technical flaws, inconsistencies, and missing elements. The review must go beyond a structured checklist and provide a nuanced, in-depth critique akin to a detailed human review.

Conceptual flaws must be explicitly discussed, questioning the coherence between objectives, claims, methodology, and experimental results.
Mathematical descriptions and formalization should be scrutinized line-by-line to check for inaccuracies, inconsistencies, or insufficient justification.
Logical reasoning gaps must be pointed out, ensuring the paper’s argumentation is sound and conclusions are well-supported.
Explicitly highlight missing sections (e.g., related work, justification for hyperparameters, ablation studies, ethical considerations).
Acknowledge strengths where present, but do not dilute the critique. The review should be direct, critical, and constructive.
Each evaluation category should be assigned a 1 to 5 score, with:

A clear justification for the score (not just general remarks).
Specific evidence from the paper (e.g., section numbers, equations, figures).
Concrete suggestions for improvement, requiring substantial effort from the authors rather than minor refinements.
📌 Evaluation Criteria
1️⃣ Originality (1-5)
Does the paper introduce a genuinely novel idea, or is it a minor variation of existing work?
Is the contribution incremental, redundant, or lacking originality?
Does the paper convincingly justify why its approach is novel?
✅ Strengths: Identify truly novel aspects (if any).
❌ Weaknesses: Directly state if the contribution is weak or insufficiently justified. If prior work already covers similar ideas, call this out explicitly.
🔹 Improvement Suggestion: Insist on stronger theoretical or empirical differentiation from prior work. If the novelty is weak, suggest alternative problem settings or methodological innovations.

2️⃣ Theoretical Soundness & Methodological Rigor (1-5)
Are theoretical claims well-supported, or are they vague and hand-wavy?
Are assumptions valid, clearly justified, and non-trivial?
Do the mathematical formalizations contain inconsistencies, ambiguities, or unstated constraints?
Are key derivations and proofs complete, or do they skip crucial steps?
✅ Strengths: If formalism is clear and rigorous, acknowledge it.
❌ Weaknesses: Identify any logical inconsistencies, missing derivations, or incorrect assumptions. If methodology lacks proper justification, call it out explicitly.
🔹 Improvement Suggestion: Demand full proof verification, correction of flawed assumptions, or additional theoretical justification.

3️⃣ Coherence Between Claims and Experiments (1-5)
Do the experimental results actually support the paper’s claims?
Are there hidden contradictions between what the paper argues and what the results demonstrate?
Does the discussion overstate the significance of results?
✅ Strengths: Highlight cases where results match claims.
❌ Weaknesses: Point out any instances of cherry-picking, over-exaggeration, or contradictions between methodology and findings.
🔹 Improvement Suggestion: Demand revised claims, stronger justification of conclusions, or additional experiments to validate assertions.

4️⃣ Experimental Soundness (1-5)
Are the experimental results statistically robust?
Are baselines appropriate, strong, and up-to-date?
Are hyperparameters, datasets, and training details properly reported?
Does the experimental design support generalizability, or are results cherry-picked?
✅ Strengths: Recognize cases where experiments are well-designed.
❌ Weaknesses: Identify weak baselines, lack of statistical tests, omitted ablation studies, or missing hyperparameter settings.
🔹 Improvement Suggestion: Suggest stronger baselines, more ablation studies, and clearer reporting of all experimental details.

5️⃣ Depth and Technical Substance (1-5)
Does the paper demonstrate deep technical insight, or is it shallow and superficial?
Is the problem approached from multiple angles, or is the exploration surface-level?
✅ Strengths: Acknowledge comprehensive exploration.
❌ Weaknesses: Call out shallow reasoning, oversimplifications, or lack of depth.
🔹 Improvement Suggestion: Suggest more rigorous technical exploration, alternative perspectives, or additional theoretical backing.

6️⃣ Clarity & Presentation (1-5)
Are key concepts well-explained, or does the paper assume too much background knowledge?
Are notation and figures clear and self-contained?
Are there ambiguities in definitions or unclear sections?
✅ Strengths: Recognize well-structured explanations.
❌ Weaknesses: Point out vague writing, inconsistent notation, or unexplained jargon.
🔹 Improvement Suggestion: Suggest rewriting unclear sections, adding more intuitive explanations, or improving figures.

7️⃣ Reproducibility & Transparency (1-5)
Can others fully reproduce the results?
Are code, datasets, and hyperparameters provided?
Are crucial experimental details missing or ambiguous?
✅ Strengths: If reproducibility is strong, acknowledge it.
❌ Weaknesses: Criticize missing details, lack of dataset/code access, or vague descriptions.
🔹 Improvement Suggestion: Demand open-source code, full dataset access, and detailed hyperparameter reporting.

8️⃣ Missing Elements (Standalone Section)
Beyond scoring, explicitly identify if any key sections are absent, such as:

Missing Related Work section.
No explanation of hyperparameters or training details.
No ablation studies or statistical significance tests.
If something critical is missing, call it out explicitly rather than relying on low scores.

📌 Conference-Specific Evaluation
Beyond standard criteria, assess the paper against specific expectations of the target conference (e.g., NeurIPS, ICLR, ACL, ICML, EMNLP).

# | **Conference** | **Additional Evaluation Areas** |
# | :---------- | :-------------------------------------------- |
# | **NeurIPS** | **Impact (15%)**, **Theoretical Depth (10%)**, **Reproducibility (5%)** |
# | **ICLR** | **Reproducibility (20%)**, **Open Science (10%)**, **Negative Results (5%)** |
# | **ACL** | **Ethics (15%)**, **Meaningful Comparison (10%)**, **Multilinguality (5%)** |
# | **ICML** | **Algorithmic Innovation (20%)**, **Scalability (10%)** |
# | **EMNLP** | **Practical Utility (20%)**, **Dataset Quality (10%)** |

# For the selected conference, provide **additional ratings and explanations** based on these **secondary criteria**.



📌 Final Recommendations
✅ Overall Verdict (1-10): Justify the final rating in clear, critical terms.
✅ Reviewer Confidence (1-5): Rate how confident you are in this evaluation.
✅ Strongest Contributions: List any strengths without diluting criticism.
✅ Critical Weaknesses: Directly state why the paper is flawed and if it is not ready for publication.
✅ How to Improve for Acceptance: Provide 3-5 major, non-trivial steps that require significant effort.


"""

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(MODEL_NAME)
        # Truncate text if too long (adjust limit based on model's requirements)
        truncated_text = text[:3000]
        # Incorporate the selected conference into the prompt
        combined_prompt = (
            f"Evaluation for conference: {conference}\n\n"
            f"{prompt}\n\n"
            f"Additional Instructions: {add_prompt}\n\n"
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
        # validate_prompt(prompt)
        
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
            evaluation = evaluate_paper(extracted_text, gemini_key,  conference,prompt)
            
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

@app.post("/compare/")
async def compare_reviews(
    gemini_key: str = Form(...),
    gemini_review: str = Form(...),
    human_review: str = Form(...)
) -> JSONResponse:
    """
    Compare the Gemini AI evaluation with a human review.
    """
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(MODEL_NAME)
        comparison_prompt = (
            """📌 Comparison Prompt
Compare the Gemini AI-generated review with the human review of the same research paper.

Identify similarities in evaluation, scoring, and critique.
Highlight differences in reasoning, focus areas, and tone.
Analyze whether both reviews agree on the paper’s strengths and weaknesses.
Check for missing elements in either review that could impact the evaluation.
Assess the clarity, depth, and accuracy of the AI-generated review in comparison to the human evaluator.
📌 Scoring Criteria (Similarity Score: 1-10)
Assign a similarity score from 1 to 10, where:

10 → Nearly identical reviews with highly similar reasoning, critique, and suggestions.
7-9 → Mostly similar, with minor differences in wording, emphasis, or depth.
4-6 → Moderately similar, but with some notable discrepancies in evaluation.
1-3 → Significantly different, with conflicting evaluations or major gaps.
Provide a final similarity score and explain the key factors influencing the rating.

Gemini AI Review:
{gemini_review}

Human Review:
{human_review}

Output Structure:

Key Similarities
Key Differences
Overall Similarity Score (1-10)
Final Conclusion on AI vs. Human Review Alignment\n\n"""
            f"**Gemini AI Review:**\n{gemini_review}\n\n"
            f"**Human Review:**\n{human_review}"
        )
        response = model.generate_content(comparison_prompt)
        if not hasattr(response, "text") or not response.text:
            raise HTTPException(
                status_code=500,
                detail="No response received from Gemini AI"
            )
        return JSONResponse(content={"comparison": response.text}, status_code=200)
    except Exception as e:
        logger.error(f"Error in Gemini AI comparison processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in AI processing: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# To run the app:
# uvicorn main:app --reload
