from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import io
import os
import json
import uuid
import glob
from datetime import datetime, timezone
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- App Setup ---
app = FastAPI(
    title="Resume Parser API",
    description="API to parse resumes, extract information, and return structured data.",
    version="1.0.0"
)

# --- Storage Setup ---
STORAGE_DIR = "analyses_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- CORS ---
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Experience(BaseModel):
    title: Optional[str] = "Not specified"
    company: Optional[str] = "Not specified"
    date: Optional[str] = "Not specified"
    description: Optional[str] = "Not specified"

class Education(BaseModel):
    degree: Optional[str] = "Not specified"
    institution: Optional[str] = "Not specified"
    date: Optional[str] = "Not specified"

class ResumeData(BaseModel):
    name: Optional[str] = "Not specified"
    email: Optional[str] = "Not specified"
    phone: Optional[str] = "Not specified"
    summary: Optional[str] = "Not specified"
    experience: List[Experience] = []
    education: List[Education] = []
    skills: List[str] = []

class ArchivedResume(ResumeData):
    id: str
    filename: str
    timestamp: str

class AnalysisStub(BaseModel):
    id: str
    filename: str
    timestamp: str

# --- LangChain Setup ---
llm = ChatOllama(model="gemma3:1b", format="json")
parser = JsonOutputParser(pydantic_object=ResumeData)
template = """
You are an expert resume parser. Based on the resume text provided below, extract the information and generate a single JSON object that strictly follows the structure provided.

**JSON STRUCTURE TO FOLLOW:**
{{
    "name": "The full name of the candidate",
    "email": "The candidate's email address",
    "phone": "The candidate's phone number",
    "summary": "A brief professional summary from the resume",
    "experience": [
        {{
            "title": "Job title",
            "company": "Company name",
            "date": "Dates of employment",
            "description": "A summary of responsibilities and achievements"
        }}
    ],
    "education": [
        {{
            "degree": "Degree or certificate name",
            "institution": "Name of the school or institution",
            "date": "Dates of attendance"
        }}
    ],
    "skills": ["A list of skills, e.g., 'Python', 'Project Management'"]
}}

**IMPORTANT RULES:**
- You MUST only respond with the single JSON object. Do not add any introductory text, explanations, or markdown formatting like ```json.
- The 'experience' and 'education' fields MUST be arrays (lists) of objects, even if only one item is found for each.
- If a specific piece of information is not found in the resume, use "Not specified" for string fields or an empty list `[]` for arrays like 'skills', 'experience', or 'education'.

**RESUME TEXT TO PARSE:**
---
{resume_text}
---

Now, provide the JSON object.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | parser

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/analyses", response_model=List[AnalysisStub])
def get_analyses_history():
    """Returns a list of all previously analyzed resumes."""
    history = []
    for filepath in glob.glob(f"{STORAGE_DIR}/*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            history.append({
                "id": data.get("id"),
                "filename": data.get("filename"),
                "timestamp": data.get("timestamp")
            })
    # Sort by timestamp, newest first
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    return history

@app.get("/analyses/{analysis_id}", response_model=ArchivedResume)
def get_analysis_detail(analysis_id: str):
    """Returns the full data for a single analysis."""
    filepath = f"{STORAGE_DIR}/{analysis_id}.json"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Analysis not found")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.post("/parse-resume", response_model=ArchivedResume)
async def parse_resume(file: UploadFile = File(...)):
    """Parses a new resume, saves it, and returns the structured data."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        pdf_content = await file.read()
        text = ""
        with io.BytesIO(pdf_content) as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        
        print("--- EXTRACTED PDF TEXT ---")
        print(text)
        print("--------------------------")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Invoke the chain
        parsed_data = await chain.ainvoke({"resume_text": text})

        print("--- PARSED DATA FROM LLM ---")
        print(parsed_data)
        print("----------------------------")

        # Create the full archive object
        new_analysis = ArchivedResume(
            id=str(uuid.uuid4()),
            filename=file.filename,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **parsed_data
        )

        # Save the new analysis to a file
        save_path = f"{STORAGE_DIR}/{new_analysis.id}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_analysis.dict(), f, indent=2)

        return new_analysis

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run the app:
# uvicorn main:app --reload 