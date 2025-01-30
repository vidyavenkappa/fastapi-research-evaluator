import os
import time
import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
GENAI_API_KEY = "your-gemini-api-key"  # Replace with your Google Gemini API Key
genai.configure(api_key=GENAI_API_KEY)

# Initialize FastAPI
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the evaluation prompt
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


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to upload file to Gemini
def upload_to_gemini(file_path):
    file = genai.upload_file(file_path, mime_type="application/pdf")
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Function to wait until files are active in Gemini API
def wait_for_files_active(files):
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

# Function to evaluate the research paper using Gemini
def evaluate_paper(file_path):
    try:
        # Upload PDF to Gemini
        files = [upload_to_gemini(file_path)]
        wait_for_files_active(files)

        # Initialize Gemini Model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel("gemini-2.0-flash-exp", generation_config=generation_config)
        
        # Start chat session
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [files[0], PROMPT],
                }
            ]
        )

        # Send evaluation request
        response = chat_session.send_message(f"{files[0]}\n\n{PROMPT}")

        return response.text

    except Exception as e:
        return f"Error in processing: {str(e)}"

# FastAPI Endpoint: Upload and Evaluate PDF
@app.post("/upload/")
async def upload_paper(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Evaluate the paper
        evaluation = evaluate_paper(file_path)

        # Cleanup
        os.remove(file_path)

        return JSONResponse(content={"evaluation": evaluation})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the server with: uvicorn main:app --reload
