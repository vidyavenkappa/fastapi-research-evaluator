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
            detail=f"File size exceeds maximum limit of {2*MAX_FILE_SIZE/1024/1024}MB"
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
#         prompt="""
# **📌 Research Paper Evaluation Prompt**

# I have uploaded a research paper in PDF format, and I need a **structured, thorough evaluation** to determine its likelihood of acceptance at a specific **ML/NLP conference** (NeurIPS, ICLR, ICML, CoNLL, ACL, EMNLP). The review should be based on **standard peer-review criteria** along with additional **conference-specific evaluation criteria**.

# The evaluation must be **specific and detailed**, referencing **actual parts of the paper** (e.g., equations, figures, tables, methodology sections). Each category should be assigned a **score from 1 to 5**, with **clear reasoning** for the score and **specific improvement suggestions**.

# ---

# ## **📌 Evaluation Criteria**

# For each criterion below, assign a **score from 1 to 5** and provide:
# - **Detailed Justification** → Why was this score assigned?  
# - **Specific Evidence** → Cite sections of the paper that support this evaluation.  
# - **Actionable Suggestions** → Provide concrete recommendations for improvement.  

# ### **1️⃣ Originality (1-5)**
# - Does the paper introduce a **novel idea, model, or approach**?
# - How does it differ from previous work?
# - Are there **clear innovations**, or is it an **incremental improvement**?

# 🔹 **Improvement Suggestion**: Differentiate the method further, add novel aspects, or compare with additional baselines.

# ### **2️⃣ Soundness & Correctness (1-5)**
# - Is the methodology logically sound and **theoretically justified**?
# - Are **assumptions valid**, and are there any **mathematical flaws**?
# - Are experiments **statistically significant**?

# 🔹 **Improvement Suggestion**: Add missing proofs, conduct ablation studies, or strengthen experimental justification.

# ### **3️⃣ Clarity (1-5)**
# - Is the paper **well-organized and easy to follow**?
# - Are technical terms, concepts, and figures **clearly explained**?

# 🔹 **Improvement Suggestion**: Rewrite unclear sections, standardize notation, or improve figure explanations.

# ### **4️⃣ Meaningful Comparison (1-5)**
# - Does the paper **compare results with prior work**?
# - Are comparisons **fair**, using **strong baselines**?

# 🔹 **Improvement Suggestion**: Add missing benchmarks, evaluate against newer models.

# ### **5️⃣ Impact (1-5)**
# - How significant is the contribution?
# - Does the paper introduce a method that can lead to **new research directions**?

# 🔹 **Improvement Suggestion**: Show real-world applicability or generalization across domains.

# ### **6️⃣ Substance (1-5)**
# - Is the **work sufficiently detailed**?
# - Does it explore **multiple aspects of the problem**?

# 🔹 **Improvement Suggestion**: Add more experiments, datasets, or analysis.

# ### **7️⃣ Replicability (1-5)**
# - Can other researchers **reproduce the results**?
# - Are **code, dataset, and hyperparameters included**?

# 🔹 **Improvement Suggestion**: Provide public code repository, dataset details, and experimental settings.

# ### **8️⃣ Appropriateness (1-5)**
# - Does the paper **align with the scope** of the selected conference?

# 🔹 **Improvement Suggestion**: If misaligned, recommend a more suitable venue.

# ### **9️⃣ Ethical Concerns (1-5)**
# - Does the paper consider **bias, fairness, or ethical risks**?

# 🔹 **Improvement Suggestion**: Add analysis of biases, discuss ethical considerations.

# ### **🔟 Relation to Prior Work (1-5)**
# - Does the paper properly **cite and position itself** within existing research?

# 🔹 **Improvement Suggestion**: Add key citations from recent literature.

# ---

# ## **📌 Conference-Specific Criteria**
# In addition to general criteria, evaluate based on **conference-specific expectations**:

# | **Conference** | **Secondary Criteria & Weighting** |
# | :---------- | :----------------------------------------------------------- |
# | **NeurIPS** | **Impact (15%)**, **Theoretical Depth (10%)**, **Reproducibility (5%)** |
# | **ICLR** | **Reproducibility (20%)**, **Open Science (10%)**, **Negative Results (5%)** |
# | **ACL** | **Ethics (15%)**, **Meaningful Comparison (10%)**, **Multilinguality (5%)** |
# | **ICML** | **Algorithmic Innovation (20%)**, **Scalability (10%)** |
# | **EMNLP** | **Practical Utility (20%)**, **Dataset Quality (10%)** |

# For the **selected conference**, provide **additional ratings and explanations** based on these **secondary criteria**.

# ---

# ## **📌 Final Recommendations**
# After scoring each category, provide:
# ✅ **Overall Score (1-5)** → Justify why this score was assigned.  
# ✅ **Reviewer Confidence (1-5)** → Rate how confident you are in this evaluation.  
# ✅ **Reasons for Acceptance** → Highlight the strongest contributions.  
# ✅ **Reasons for Rejection** → Identify critical weaknesses.  
# ✅ **How to Improve for Acceptance** → Provide clear suggestions for revision.  

# ---

# ## **📌 Example Review Output**
# **Originality: 3/5**  
# ✅ **Strength:** The paper proposes a transformer-based approach for multilingual text classification (**Section 3.2, Figure 5**).  
# ❌ **Weakness:** It mostly builds on **BERT-based models** (**Section 2.1, Related Work**).  
# 🔹 **Improvement Suggestion:** Compare against **XLM-R and T5 models**.

# **Final Score: 3.5/5 (Borderline Accept)**  
# 🔹 **Suggested Improvements for Acceptance:**  
# 1. Add a **new experimental baseline (T5)** to strengthen comparisons.  
# 2. Improve **clarity in Section 3**.  
# 3. Provide **code and hyperparameter settings** for reproducibility.  

# ---

# ## **📌 UI Inputs**
# - **Conference Name**: {conference}
# - **Additional Comments**: {add_prompt}

# ### **Deliverables:**
# - **Score Breakdown (1-5) for Each Criterion**
# - **Conference-Specific Secondary Evaluation**
# - **Reason for Acceptance/Rejection**
# - **Final Recommendation** (Accept, Reject, Borderline)
# - **Reviewer Confidence (1-5)**
# - **Final Score (1-5)**

# """
        prompt = """
## **📌 Research Paper Evaluation Prompt (Refined for Human-Like Feedback)**

**Context:**  
I have uploaded a **research paper (PDF format)** for evaluation. The goal is to provide a **structured, in-depth review** assessing its **likelihood of acceptance** at a specific **ML/NLP conference** (NeurIPS, ICLR, ICML, ACL, EMNLP, CoNLL). The review should be **thorough, insightful, and actionable**, helping the author understand **strengths, weaknesses, and improvements**.

The evaluation should follow **standard peer-review criteria** while incorporating **conference-specific expectations**. The feedback should be **specific**, referencing actual parts of the paper (e.g., equations, figures, methodology) rather than generic remarks.  

Each category should be assigned a **score from 1 to 5**, with a **clear explanation** of the score, **specific evidence** from the paper, and **practical suggestions for improvement**.

---

## **📌 Evaluation Criteria**

Each criterion below should be evaluated on a **1 to 10 scale**, with **thoughtful justifications and direct references to the paper**.

### **1️⃣ Originality (1-5)**
- **Does the paper introduce a truly novel idea, model, or approach?**  
- How does it **differ from prior work**?  
- Is the contribution **incremental or groundbreaking**?  

✅ **Strengths:** Highlight novel aspects and contributions.  
❌ **Weaknesses:** Point out any areas that lack novelty or are derivative of prior work.  
🔹 **Improvement Suggestion:** Suggest ways to differentiate the approach, introduce new elements, or benchmark against stronger baselines.

---

### **2️⃣ Soundness & Correctness (1-5)**
- **Is the methodology logically sound and theoretically justified?**  
- Are assumptions **valid and reasonable**?  
- Are there any **mathematical flaws, inconsistencies, or missing justifications**?  
- Are experimental results **statistically significant and well-supported**?  

✅ **Strengths:** Identify solid theoretical contributions and well-structured methodologies.  
❌ **Weaknesses:** Highlight questionable assumptions, missing justifications, or errors.  
🔹 **Improvement Suggestion:** Recommend additional experiments, better statistical analysis, or more rigorous theoretical justifications.

---

### **3️⃣ Clarity & Presentation (1-5)**
- **Is the paper well-organized, readable, and easy to follow?**  
- Are key concepts and results clearly explained?  
- Are **notation, terminology, and figures well-presented**?  

✅ **Strengths:** Acknowledge clear writing, well-structured explanations, and strong visual aids.  
❌ **Weaknesses:** Point out confusing sections, unclear explanations, or poor figure labeling.  
🔹 **Improvement Suggestion:** Suggest specific rewrites, better structuring, or clearer figure captions.

---

### **4️⃣ Meaningful Comparison (1-5)**
- **Does the paper properly compare with prior work?**  
- Are baselines **appropriate, strong, and up-to-date**?  
- Is the evaluation **fair and justified**?  

✅ **Strengths:** Highlight fair and comprehensive comparisons.  
❌ **Weaknesses:** Point out missing benchmarks, cherry-picked results, or unfair comparisons.  
🔹 **Improvement Suggestion:** Recommend additional comparisons, stronger baselines, or more detailed analysis of results.

---

### **5️⃣ Impact & Significance (1-5)**
- **Does the work have the potential to advance research or practical applications?**  
- Does it open up **new research directions**?  

✅ **Strengths:** Highlight aspects that could influence future research or industry.  
❌ **Weaknesses:** Identify if the work is too incremental or lacks a clear impact.  
🔹 **Improvement Suggestion:** Suggest ways to demonstrate impact through more diverse experiments, real-world validation, or broader discussions.

---

### **6️⃣ Technical Depth & Substance (1-5)**
- **Is the work detailed enough to be meaningful?**  
- Does it explore multiple perspectives of the problem?  

✅ **Strengths:** Acknowledge well-explored problems, deep insights, or extensive experiments.  
❌ **Weaknesses:** Point out missing details, shallow exploration, or oversimplifications.  
🔹 **Improvement Suggestion:** Recommend additional experiments, analysis, or discussions.

---

### **7️⃣ Reproducibility (1-5)**
- **Can the results be replicated by others?**  
- Are **code, datasets, and hyperparameters included**?  

✅ **Strengths:** If reproducibility is high, highlight well-documented experiments.  
❌ **Weaknesses:** If key details are missing, point them out.  
🔹 **Improvement Suggestion:** Encourage sharing code, data, or clearer experiment details.

---

### **8️⃣ Conference Appropriateness (1-5)**
- **Is the paper aligned with the focus of the conference?**  
- Would it be better suited for another venue?  

✅ **Strengths:** Confirm alignment with conference themes.  
❌ **Weaknesses:** If misaligned, suggest a better venue.  
🔹 **Improvement Suggestion:** Adjust framing to better fit the conference.

---

### **9️⃣ Ethical Considerations (1-5)**
- **Does the paper discuss potential biases, fairness, or ethical implications?**  

✅ **Strengths:** If addressed well, acknowledge the ethical considerations.  
❌ **Weaknesses:** Highlight any overlooked ethical concerns.  
🔹 **Improvement Suggestion:** Recommend bias analysis, ethical discussions, or impact statements.

---

### **🔟 Relation to Prior Work (1-5)**
- **Does the paper properly cite and position itself within existing research?**  

✅ **Strengths:** Acknowledge thorough literature review and citations.  
❌ **Weaknesses:** If key references are missing, point them out.  
🔹 **Improvement Suggestion:** Recommend additional citations or better positioning within the literature.

---

## **📌 Conference-Specific Evaluation**
Beyond standard criteria, assess the paper against **specific expectations of the target conference**:

| **Conference** | **Additional Evaluation Areas** |
| :---------- | :-------------------------------------------- |
| **NeurIPS** | **Impact (15%)**, **Theoretical Depth (10%)**, **Reproducibility (5%)** |
| **ICLR** | **Reproducibility (20%)**, **Open Science (10%)**, **Negative Results (5%)** |
| **ACL** | **Ethics (15%)**, **Meaningful Comparison (10%)**, **Multilinguality (5%)** |
| **ICML** | **Algorithmic Innovation (20%)**, **Scalability (10%)** |
| **EMNLP** | **Practical Utility (20%)**, **Dataset Quality (10%)** |

For the selected conference, provide **additional ratings and explanations** based on these **secondary criteria**.

---

## **📌 Final Recommendations**
After scoring each category, provide a **final verdict**:

✅ **Overall Score (1-10):** Justify why this score was assigned.  
✅ **Reviewer Confidence (1-5):** Rate confidence in this evaluation.  
✅ **Strongest Contributions:** Highlight the most impressive aspects.  
✅ **Critical Weaknesses:** Identify key areas for improvement.  
✅ **How to Improve for Acceptance:** Provide **3-5 actionable steps**.  

---

## **📌 Example Review Output**
**Originality: 3/5**  
✅ **Strength:** The proposed method extends transformer-based architectures for multilingual text classification (**Section 3.2, Figure 5**).  
❌ **Weakness:** The innovation is incremental; it builds on prior BERT models without significant new contributions (**Section 2.1, Related Work**).  
🔹 **Improvement Suggestion:** Compare against **XLM-R and T5 models** to strengthen the novelty claim.

**Final Score: 7/10 (Borderline Accept)**  
🔹 **Suggested Improvements:**  
1. Include **a stronger experimental baseline** (e.g., T5).  
2. Improve **clarity in Section 3**.  
3. Provide **code and hyperparameter settings**.  

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
