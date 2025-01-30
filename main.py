import os
import time
import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Configure Gemini API

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://review-guide-frontend.vercel.app", "http://localhost:3000"],  # Allow both local & deployed frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ✅ Define the evaluation prompt
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

# ✅ Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip() if text else "Error: Unable to extract text from PDF."
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Function to evaluate the research paper using Gemini
def evaluate_paper(text, gemini_key):
    try:
        # ✅ Configure Gemini API with user-provided key
        genai.configure(api_key=gemini_key)

        # ✅ Initialize Gemini Model
        model = genai.GenerativeModel("gemini-2.0-pro")

        # ✅ Generate AI Response
        response = model.generate_content(f"{PROMPT}\n\n{text[:3000]}")  # Limit input to avoid exceeding model limits

        return response.text if hasattr(response, "text") else "Error: No response from AI."

    except Exception as e:
        return f"Error in processing: {str(e)}"

# ✅ FastAPI API Endpoint to Upload & Evaluate PDF
@app.post("/upload/")
async def upload_paper(file: UploadFile = File(...), gemini_key: str = Form(...)):
    try:
        # ✅ Save file temporarily
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # ✅ Extract text from PDF
        extracted_text = extract_text_from_pdf(file_path)

        if extracted_text.startswith("Error:"):
            return JSONResponse(content={"error": extracted_text}, status_code=400)

        # ✅ Evaluate the paper using the provided Gemini API Key
        evaluation = evaluate_paper(extracted_text, gemini_key)

        # ✅ Cleanup the temporary file
        os.remove(file_path)

        return JSONResponse(content={"evaluation": evaluation})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ Run the server with: uvicorn main:app --reload
