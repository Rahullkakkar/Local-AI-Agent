from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_vector_store
from config import Config, PRESETS, list_presets
from typing import List, Dict, Any, Optional
import os
import tempfile
from pathlib import Path

app = FastAPI(title="Universal Document Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_config = Config()
vector_store = get_vector_store(current_config)
model = OllamaLLM(model=current_config.LLM_MODEL)


class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = None  


class ChatResponse(BaseModel):
    answer: str
    relevant_documents: List[Dict[str, Any]]
    stats: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    app_name: Optional[str] = None
    app_description: Optional[str] = None
    system_prompt: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    retrieval_k: Optional[int] = None


class TextUploadRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """
    Ask a question and get an AI-generated answer based on documents
    Works with any document type - completely domain-agnostic
    """
    try:
        question = request.question
        k = request.k or current_config.RETRIEVAL_K
        
        relevant_docs = vector_store.search(question, k=k)
        
        clean_context = "\n\n".join(
            doc.page_content for doc in relevant_docs 
)
        
        prompt_template = ChatPromptTemplate.from_template(current_config.SYSTEM_PROMPT)
        chain = prompt_template | model
        
        result = chain.invoke({
            "context": clean_context,
            "question": question
        })
        
        docs_list = []
        for doc in relevant_docs:
            docs_list.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        stats = vector_store.get_stats()
        
        return ChatResponse(
            answer=result,
            relevant_documents=docs_list,
            stats=stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload any supported document file (CSV, TXT, MD, JSON, PDF)
    Automatically detects file type and processes accordingly
    """
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in current_config.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {current_config.SUPPORTED_FORMATS}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            num_docs = vector_store.add_documents_from_file(tmp_path)
            
            return {
                "message": f"Successfully processed {file.filename}",
                "documents_added": num_docs,
                "file_type": file_ext,
                "stats": vector_store.get_stats()
            }
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-text")
async def upload_text(request: TextUploadRequest):
    """
    Upload text directly (no file needed)
    Useful for pasting content, notes, or programmatic uploads
    """
    try:
        num_docs = vector_store.add_documents_from_text(
            text=request.text,
            metadata=request.metadata
        )
        
        return {
            "message": "Text uploaded successfully",
            "documents_added": num_docs,
            "stats": vector_store.get_stats()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get statistics about the document collection"""
    try:
        return vector_store.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return current_config.to_dict()


@app.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """
    Update configuration dynamically
    Changes app behavior without restart
    """
    global current_config, vector_store, model
    
    try:
        if request.app_name:
            current_config.APP_NAME = request.app_name
        if request.app_description:
            current_config.APP_DESCRIPTION = request.app_description
        if request.system_prompt:
            current_config.SYSTEM_PROMPT = request.system_prompt
        if request.retrieval_k:
            current_config.RETRIEVAL_K = request.retrieval_k
        
        if request.llm_model and request.llm_model != current_config.LLM_MODEL:
            current_config.LLM_MODEL = request.llm_model
            model = OllamaLLM(model=current_config.LLM_MODEL)
        
        if request.embedding_model and request.embedding_model != current_config.EMBEDDING_MODEL:
            current_config.EMBEDDING_MODEL = request.embedding_model
            vector_store = get_vector_store(current_config)
        
        return {
            "message": "Configuration updated successfully",
            "config": current_config.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
async def get_presets():
    """Get all available configuration presets"""
    return {
        "presets": list_presets(),
        "details": PRESETS
    }


@app.post("/api/preset/{preset_name}")
async def apply_preset(preset_name: str):
    """
    Apply a configuration preset
    Presets: reviews, legal, finance, tech, notes, research
    """
    global current_config, vector_store, model
    
    if preset_name not in PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' not found. Available: {list_presets()}"
        )
    
    try:
        preset = PRESETS[preset_name]
        
        current_config.APP_NAME = preset.get("app_name", current_config.APP_NAME)
        current_config.APP_DESCRIPTION = preset.get("app_description", current_config.APP_DESCRIPTION)
        current_config.SYSTEM_PROMPT = preset.get("system_prompt", current_config.SYSTEM_PROMPT)
        current_config.COLLECTION_NAME = preset.get("collection_name", current_config.COLLECTION_NAME)
        
        vector_store = get_vector_store(current_config)
        
        return {
            "message": f"Applied preset: {preset_name}",
            "config": current_config.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        vector_store.clear()
        return {
            "message": "All documents cleared successfully",
            "stats": vector_store.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "config": current_config.to_dict(),
        "stats": vector_store.get_stats()
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": current_config.APP_NAME,
        "description": current_config.APP_DESCRIPTION,
        "version": "2.0.0",
        "endpoints": {
            "chat": "POST /api/chat",
            "upload_file": "POST /api/upload-file",
            "upload_text": "POST /api/upload-text",
            "stats": "GET /api/stats",
            "config": "GET /api/config",
            "update_config": "POST /api/config",
            "presets": "GET /api/presets",
            "apply_preset": "POST /api/preset/{preset_name}",
            "clear_documents": "DELETE /api/documents",
            "health": "GET /api/health"
        },
        "supported_formats": current_config.SUPPORTED_FORMATS
    }


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting {current_config.APP_NAME}")
    print(f"📝 {current_config.APP_DESCRIPTION}")
    print(f"🔧 Model: {current_config.LLM_MODEL}")
    print(f"📚 Supported formats: {current_config.SUPPORTED_FORMATS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

