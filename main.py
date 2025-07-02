"""
RESUME PARSER API - BACKEND
===========================

Este é o backend principal do sistema Resume Parser, construído com FastAPI.
O sistema permite upload de PDFs de currículos e extrai informações estruturadas
usando inteligência artificial através do LangChain e modelo Ollama.

Funcionalidades principais:
- Upload e processamento de arquivos PDF
- Extração de texto usando PyPDF2
- Processamento de IA usando LangChain + Ollama
- Armazenamento local em arquivos JSON
- API RESTful com endpoints para CRUD de análises

Tecnologias utilizadas:
- FastAPI: Framework web moderno para APIs
- LangChain: Framework para aplicações com LLM
- Ollama: Servidor local de modelos de linguagem
- PyPDF2: Biblioteca para processamento de PDFs
- Pydantic: Validação e serialização de dados
"""

# Importações das bibliotecas principais
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2  # Para extração de texto de PDFs
import io  # Para manipulação de arquivos em memória
import os  # Para operações do sistema de arquivos
import json  # Para persistência de dados
import uuid  # Para geração de IDs únicos
import glob  # Para busca de arquivos
from datetime import datetime, timezone  # Para timestamps
from langchain_community.chat_models import ChatOllama  # Integração com Ollama
from langchain_core.output_parsers import JsonOutputParser  # Parser de JSON
from langchain_core.prompts import ChatPromptTemplate  # Templates de prompt

# =============================================================================
# CONFIGURAÇÃO DA APLICAÇÃO
# =============================================================================

# Criação da instância principal do FastAPI
# Define metadados da API que aparecem na documentação automática (/docs)
app = FastAPI(
    title="Resume Parser API",
    description="API to parse resumes, extract information, and return structured data.",
    version="1.0.0"
)

# =============================================================================
# CONFIGURAÇÃO DE ARMAZENAMENTO
# =============================================================================

# Diretório onde serão armazenados os arquivos JSON das análises
# Cada análise processada é salva como um arquivo JSON único
STORAGE_DIR = "analyses_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)  # Cria o diretório se não existir

# =============================================================================
# CONFIGURAÇÃO DE CORS (Cross-Origin Resource Sharing)
# =============================================================================

# Lista de origens permitidas para requisições do frontend
# Necessário para permitir que o React (localhost:5173) acesse a API
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Configuração do middleware CORS
# Permite requisições entre diferentes origens (frontend React + backend FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Apenas essas origens podem fazer requisições
    allow_credentials=True,  # Permite envio de cookies e headers de autenticação
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

# =============================================================================
# MODELOS DE DADOS (PYDANTIC)
# =============================================================================

# Os modelos Pydantic definem a estrutura e validação dos dados
# Eles são usados para:
# 1. Validar dados de entrada e saída da API
# 2. Gerar documentação automática da API
# 3. Serialização/deserialização JSON
# 4. Type hints para melhor desenvolvimento

class Experience(BaseModel):
    """
    Modelo para representar uma experiência profissional
    
    Campos:
    - title: Cargo/posição ocupada
    - company: Nome da empresa
    - date: Período de trabalho (formato livre)
    - description: Descrição das responsabilidades e conquistas
    """
    title: Optional[str] = "Not specified"
    company: Optional[str] = "Not specified" 
    date: Optional[str] = "Not specified"
    description: Optional[str] = "Not specified"

class Education(BaseModel):
    """
    Modelo para representar formação educacional
    
    Campos:
    - degree: Nome do curso/diploma/certificação
    - institution: Nome da instituição de ensino
    - date: Período de estudo (formato livre)
    """
    degree: Optional[str] = "Not specified"
    institution: Optional[str] = "Not specified"
    date: Optional[str] = "Not specified"

class ResumeData(BaseModel):
    """
    Modelo principal que representa todos os dados extraídos de um currículo
    
    Este é o modelo base que contém toda a informação estruturada:
    - Dados pessoais (nome, email, telefone)
    - Resumo profissional
    - Lista de experiências profissionais
    - Lista de formações educacionais
    - Lista de habilidades/competências
    """
    name: Optional[str] = "Not specified"
    email: Optional[str] = "Not specified"
    phone: Optional[str] = "Not specified"
    summary: Optional[str] = "Not specified"
    experience: List[Experience] = []  # Lista de experiências profissionais
    education: List[Education] = []    # Lista de formações educacionais
    skills: List[str] = []            # Lista de habilidades

class ArchivedResume(ResumeData):
    """
    Modelo para currículo arquivado/persistido
    
    Herda todos os campos de ResumeData e adiciona metadados:
    - id: Identificador único (UUID)
    - filename: Nome do arquivo original
    - timestamp: Data/hora do processamento
    
    Este modelo é usado para salvar análises no sistema de arquivos
    e retornar dados completos com metadados para o frontend
    """
    id: str          # UUID único gerado automaticamente
    filename: str    # Nome do arquivo PDF original
    timestamp: str   # ISO timestamp do momento da análise

class AnalysisStub(BaseModel):
    """
    Modelo simplificado para listagem de análises
    
    Usado no endpoint de histórico (/analyses) para retornar apenas
    informações essenciais sem carregar todos os dados do currículo.
    Otimiza performance e reduz transferência de dados.
    """
    id: str          # UUID da análise
    filename: str    # Nome do arquivo original
    timestamp: str   # Timestamp da análise

# =============================================================================
# CONFIGURAÇÃO DO LANGCHAIN E INTELIGÊNCIA ARTIFICIAL
# =============================================================================

# Configuração do modelo de linguagem local (Ollama)
# gemma3:1b é um modelo pequeno e eficiente para extração de dados
# O parâmetro format="json" força o modelo a retornar JSON válido
llm = ChatOllama(model="gemma3:1b", format="json")

# Parser que converte a saída JSON do modelo para objetos Pydantic
# Garante que a estrutura retornada seja compatível com ResumeData
parser = JsonOutputParser(pydantic_object=ResumeData)

# Template do prompt que instrui o modelo sobre como extrair informações
# Este prompt é crítico para a qualidade da extração de dados
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
- Be extremely careful with spelling and numbers. Extract information as accurately as possible, preferring to copy it verbatim.
- If a specific piece of information is not found in the resume, use "Not specified" for string fields or an empty list `[]` for arrays like 'skills', 'experience', or 'education'.

**RESUME TEXT TO PARSE:**
---
{resume_text}
---

Now, provide the JSON object.
"""

# Criação do prompt template a partir do template string
prompt = ChatPromptTemplate.from_template(template)

# Criação da cadeia de processamento (chain)
# prompt -> llm -> parser
# 1. Prompt formatado é enviado para o modelo
# 2. Modelo processa e retorna JSON
# 3. Parser converte JSON para objeto Pydantic
chain = prompt | llm | parser

# =============================================================================
# ENDPOINTS DA API
# =============================================================================

@app.get("/")
def read_root():
    """
    ENDPOINT DE HEALTH CHECK
    
    Endpoint simples para verificar se a API está funcionando.
    Usado para monitoramento e verificação de status do serviço.
    
    Retorna:
        dict: Status da API
    """
    return {"status": "API is running"}

@app.get("/analyses", response_model=List[AnalysisStub])
def get_analyses_history():
    """
    ENDPOINT PARA LISTAR HISTÓRICO DE ANÁLISES
    
    Retorna uma lista com informações básicas de todas as análises
    já processadas e armazenadas no sistema.
    
    Processo:
    1. Escaneia o diretório de armazenamento
    2. Lê metadados de cada arquivo JSON
    3. Retorna lista ordenada por timestamp (mais recente primeiro)
    
    Retorna:
        List[AnalysisStub]: Lista de análises com metadados básicos
    
    Erros:
        - Problemas de sistema de arquivos são tratados silenciosamente
        - Arquivos JSON inválidos são ignorados
    """
    history = []
    
    # Busca todos os arquivos JSON no diretório de armazenamento
    for filepath in glob.glob(f"{STORAGE_DIR}/*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extrai apenas metadados necessários para a listagem
                history.append({
                    "id": data.get("id"),
                    "filename": data.get("filename"),
                    "timestamp": data.get("timestamp")
                })
        except (json.JSONDecodeError, KeyError, IOError):
            # Ignora arquivos corrompidos ou ilegíveis
            continue
    
    # Ordena por timestamp, mais recente primeiro
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    return history

@app.get("/analyses/{analysis_id}", response_model=ArchivedResume)
def get_analysis_detail(analysis_id: str):
    """
    ENDPOINT PARA OBTER DETALHES DE UMA ANÁLISE ESPECÍFICA
    
    Retorna todos os dados completos de uma análise baseado no ID.
    Usado quando o usuário seleciona uma análise específica no histórico.
    
    Parâmetros:
        analysis_id (str): UUID da análise desejada
    
    Retorna:
        ArchivedResume: Dados completos da análise incluindo:
            - Informações pessoais extraídas
            - Experiências profissionais
            - Formação educacional
            - Habilidades
            - Metadados (ID, filename, timestamp)
    
    Erros:
        HTTPException 404: Análise não encontrada
    """
    # Constrói o caminho do arquivo baseado no ID
    filepath = f"{STORAGE_DIR}/{analysis_id}.json"
    
    # Verifica se o arquivo existe
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, 
            detail="Analysis not found"
        )
    
    # Carrega e retorna os dados completos
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading analysis data: {str(e)}"
        )

@app.post("/parse-resume", response_model=ArchivedResume)
async def parse_resume(file: UploadFile = File(...)):
    """
    ENDPOINT PRINCIPAL PARA PROCESSAMENTO DE CURRÍCULOS
    
    Este é o endpoint mais complexo e importante da API. Ele recebe um arquivo PDF,
    extrai o texto, processa com IA e retorna dados estruturados.
    
    Processo completo:
    1. Validação do tipo de arquivo (apenas PDF)
    2. Extração de texto usando PyPDF2
    3. Processamento com LangChain + Ollama
    4. Criação de objeto ArchivedResume com metadados
    5. Persistência em arquivo JSON
    6. Retorno dos dados estruturados
    
    Parâmetros:
        file (UploadFile): Arquivo PDF do currículo
    
    Retorna:
        ArchivedResume: Dados completos extraídos e processados
    
    Erros:
        HTTPException 400: Tipo de arquivo inválido ou PDF não legível
        HTTPException 500: Erro interno no processamento
    """
    
    # =============================================================================
    # VALIDAÇÃO DO ARQUIVO
    # =============================================================================
    
    # Verifica se o arquivo é um PDF
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a PDF."
        )

    try:
        # =============================================================================
        # EXTRAÇÃO DE TEXTO DO PDF
        # =============================================================================
        
        # Lê o conteúdo do arquivo em memória
        pdf_content = await file.read()
        text = ""
        
        # Processa o PDF usando PyPDF2
        with io.BytesIO(pdf_content) as f:
            reader = PyPDF2.PdfReader(f)
            
            # Extrai texto de todas as páginas
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        
        # Log do texto extraído para debug
        print("--- EXTRACTED PDF TEXT ---")
        print(text)
        print("--------------------------")
        
        # Verifica se conseguiu extrair texto
        if not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF."
            )

        # =============================================================================
        # PROCESSAMENTO COM INTELIGÊNCIA ARTIFICIAL
        # =============================================================================
        
        # Invoca a cadeia LangChain para processar o texto
        # O texto é enviado para o modelo Ollama que extrai informações estruturadas
        parsed_data = await chain.ainvoke({"resume_text": text})

        # Log dos dados processados para debug
        print("--- PARSED DATA FROM LLM ---")
        print(parsed_data)
        print("----------------------------")

        # =============================================================================
        # CRIAÇÃO DO OBJETO ARQUIVO COMPLETO
        # =============================================================================
        
        # Cria objeto ArchivedResume com dados extraídos + metadados
        new_analysis = ArchivedResume(
            id=str(uuid.uuid4()),  # Gera UUID único
            filename=file.filename,  # Nome do arquivo original
            timestamp=datetime.now(timezone.utc).isoformat(),  # Timestamp ISO
            **parsed_data  # Dados extraídos pela IA
        )

        # =============================================================================
        # PERSISTÊNCIA DOS DADOS
        # =============================================================================
        
        # Salva a análise em arquivo JSON
        save_path = f"{STORAGE_DIR}/{new_analysis.id}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            # Converte para dict e salva com indentação para legibilidade
            json.dump(new_analysis.dict(), f, indent=2, ensure_ascii=False)

        return new_analysis

    except HTTPException:
        # Re-levanta HTTPExceptions já tratadas
        raise
    except Exception as e:
        # Captura qualquer erro não previsto
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

# =============================================================================
# INSTRUÇÃO PARA EXECUÇÃO
# =============================================================================
# Para executar a aplicação em modo de desenvolvimento:
# uvicorn main:app --reload
#
# A API estará disponível em:
# - http://localhost:8000 (aplicação)
# - http://localhost:8000/docs (documentação automática) 