import json
import os



class Config:
    """
    Universal configuration that adapts to any document type or domain
    """
    
    APP_NAME = os.getenv("APP_NAME", "Document Assistant")
    APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Your AI-powered document assistant")
    
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    DB_LOCATION = os.getenv("DB_LOCATION", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))  
    
    SYSTEM_PROMPT = os.getenv(
        "SYSTEM_PROMPT",
        """You are an intelligent assistant helping users understand and analyze their documents.

Here are relevant document excerpts: {context}

User question: {question}

Provide a helpful, accurate answer based on the documents provided. If the documents don't contain relevant information, say so clearly."""
    )
    
    CSV_CONTENT_COLUMNS = os.getenv("CSV_CONTENT_COLUMNS", "").split(",") if os.getenv("CSV_CONTENT_COLUMNS") else None
    CSV_METADATA_COLUMNS = os.getenv("CSV_METADATA_COLUMNS", "").split(",") if os.getenv("CSV_METADATA_COLUMNS") else None
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  
    
    SUPPORTED_FORMATS = ['.csv', '.txt', '.md', '.json', '.pdf']
    
    @classmethod
    def to_dict(cls):
        """Export config as dictionary"""
        return {
            "app_name": cls.APP_NAME,
            "app_description": cls.APP_DESCRIPTION,
            "llm_model": cls.LLM_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "db_location": cls.DB_LOCATION,
            "collection_name": cls.COLLECTION_NAME,
            "retrieval_k": cls.RETRIEVAL_K,
            "supported_formats": cls.SUPPORTED_FORMATS
        }
    
    @classmethod
    def save_preset(cls, preset_name, config_dict):
        """Save a configuration preset"""
        presets_dir = "./config_presets"
        os.makedirs(presets_dir, exist_ok=True)
        
        with open(f"{presets_dir}/{preset_name}.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_preset(cls, preset_name):
        """Load a configuration preset"""
        preset_file = f"./config_presets/{preset_name}.json"
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                return json.load(f)
        return None



PRESETS = {
    "reviews": {
        "app_name": "Review Assistant",
        "app_description": "Analyze customer reviews and feedback",
        "system_prompt": """You are an expert at analyzing customer reviews and feedback.

Relevant reviews: {context}

Question: {question}

Provide insights about customer sentiment, common themes, and specific feedback.""",
        "collection_name": "reviews"
    },
    
    "legal": {
        "app_name": "Legal Document Assistant",
        "app_description": "Search and analyze legal documents",
        "system_prompt": """You are a legal document assistant helping users find information in legal texts.

Relevant document sections: {context}

Question: {question}

Provide accurate information based on the documents. Always cite which document the information comes from.""",
        "collection_name": "legal_docs"
    },
    
    "finance": {
        "app_name": "Financial Document Analyst",
        "app_description": "Analyze financial reports and data",
        "system_prompt": """You are a financial analyst helping users understand financial documents.

Relevant financial data: {context}

Question: {question}

Provide clear analysis and highlight key financial metrics or trends.""",
        "collection_name": "financial_docs"
    },
    
    "tech": {
        "app_name": "Technical Documentation Helper",
        "app_description": "Search technical documentation and guides",
        "system_prompt": """You are a technical documentation assistant.

Relevant documentation: {context}

Question: {question}

Provide clear, technical answers with code examples when relevant.""",
        "collection_name": "tech_docs"
    },
    
    "notes": {
        "app_name": "Personal Knowledge Base",
        "app_description": "Search your personal notes and knowledge",
        "system_prompt": """You are a personal knowledge assistant helping retrieve information from notes.

Relevant notes: {context}

Question: {question}

Help the user find and understand information from their notes.""",
        "collection_name": "personal_notes"
    },
    
    "research": {
        "app_name": "Research Paper Assistant",
        "app_description": "Search and analyze research papers",
        "system_prompt": """You are a research assistant helping analyze academic papers.

Relevant research excerpts: {context}

Question: {question}

Provide scholarly insights and cite sources when possible.""",
        "collection_name": "research_papers"
    }
}


def get_preset(name):
    """Get a preset configuration by name"""
    return PRESETS.get(name, {})


def list_presets():
    """List all available presets"""
    return list(PRESETS.keys())
