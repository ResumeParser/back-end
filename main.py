from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import io

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(
    title="Resume Parser API",
    description="API to parse resumes, extract information, and return structured data.",
    version="1.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",  # Default Vite dev server port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models based on front-end's ResumeData interface
class Experience(BaseModel):
    title: str
    company: str
    date: str
    description: str

class Education(BaseModel):
    degree: str
    institution: str
    date: str

class ResumeData(BaseModel):
    name: str
    email: str
    phone: str
    summary: str
    experience: List[Experience]
    education: List[Education]
    skills: List[str]

# 1. Initialize the LLM and the Parser
llm = ChatOllama(model="gemma3:1b", format="json")
parser = JsonOutputParser(pydantic_object=ResumeData)

# 2. Create the Prompt Template
template = """
You are an expert in parsing resumes. Your task is to extract information from the provided resume text and format it into a structured JSON object.

Based on the following resume text, extract the information and format it according to the provided JSON schema.

**JSON Schema:**
{schema}

**Resume Text:**
---
{resume_text}
---

**Instructions:**
- The 'name' should be the full name of the candidate.
- The 'email' should be a valid email address.
- The 'phone' should be a valid phone number.
- The 'summary' should be a concise professional summary. If not present, create a brief summary (1-2 sentences) based on the most recent job title and key skills.
- The 'experience' section should be a list of jobs. Each job must include a title, company, date, and description.
- The 'education' section should be a list of degrees. Each degree must include the degree name, institution, and date.
- The 'skills' section should be a list of relevant technical and soft skills.
- If a specific piece of information is not found, use a sensible default like "Not specified".
"""
prompt = ChatPromptTemplate.from_template(template, partial_variables={"schema": parser.get_format_instructions()})

# 3. Create the Chain
chain = prompt | llm | parser

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Invoke the chain with the extracted text
        response = await chain.ainvoke({"resume_text": text})
        
        return response

    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run the app:
# uvicorn main:app --reload 