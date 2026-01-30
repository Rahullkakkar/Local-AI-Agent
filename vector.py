from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from config import Config

EMBED_MAX_CHARS = 512


class UniversalVectorStore:
    """
    Universal vector store that handles any document type:
    CSV, TXT, MD, JSON, PDF, etc.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL,
        num_ctx = 2048)
        
        self.vector_store = Chroma(
            collection_name=self.config.COLLECTION_NAME,
            persist_directory=self.config.DB_LOCATION,
            embedding_function=self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def process_csv(self, file_path: str) -> List[Document]:
        """Process CSV files - auto-detects columns"""
        df = pd.read_csv(file_path)
        documents = []
        
        if self.config.CSV_CONTENT_COLUMNS:
            content_cols = self.config.CSV_CONTENT_COLUMNS
        else:
            content_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if self.config.CSV_METADATA_COLUMNS:
            metadata_cols = self.config.CSV_METADATA_COLUMNS
        else:
            metadata_cols = [col for col in df.columns if col not in content_cols]
        
        for idx, row in df.iterrows():
            content_parts = []
            for col in content_cols:
                if pd.notna(row[col]):
                    content_parts.append(f"{col}: {row[col]}")
            
            page_content = "\n".join(content_parts)
            
            metadata = {"source": file_path, "row_id": idx}
            for col in metadata_cols:
                if pd.notna(row[col]):
                    metadata[col] = row[col]
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        return documents
    
    def process_text_file(self, file_path: str) -> List[Document]:
        """Process TXT/MD files with chunking"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self.text_splitter.split_text(text)

        safe_chunks = [
                    chunk[:EMBED_MAX_CHARS]
                    for chunk in chunks
                    if chunk and chunk.strip()
]
        
        documents = []
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk_id": idx,
                    "file_type": Path(file_path).suffix
                }
            ))
        
        return documents
    
    def process_json(self, file_path: str) -> List[Document]:
        """Process JSON files - handles arrays and objects"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    content = json.dumps(item, indent=2)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "item_id": idx,
                            **{k: v for k, v in item.items() if isinstance(v, (str, int, float, bool))}
                        }
                    ))
                else:
                    documents.append(Document(
                        page_content=str(item),
                        metadata={"source": file_path, "item_id": idx}
                    ))
        
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2)
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path, "type": "json_object"}
            ))
        
        return documents
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files"""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                text = text[:100_000]
                
                chunks = self.text_splitter.split_text(text)

                safe_chunks = [
                    chunk[:EMBED_MAX_CHARS]
                    for chunk in chunks
                    if chunk and chunk.strip()
]
                
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "chunk_id": chunk_idx,
                            "file_type": "pdf"
                        }
                    ))
            
            return documents
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")
    
    def add_documents_from_file(self, file_path: str) -> int:
        """
        Universal document processor - detects file type and processes accordingly
        Returns: number of documents added
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            documents = self.process_csv(file_path)
        elif file_ext in ['.txt', '.md']:
            documents = self.process_text_file(file_path)
        elif file_ext == '.json':
            documents = self.process_json(file_path)
        elif file_ext == '.pdf':
            documents = self.process_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        existing_count = len(self.vector_store.get()["ids"]) if self.vector_store.get()["ids"] else 0
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        for doc, doc_id in zip(documents, ids):
            self.vector_store.add_documents(
                documents=[doc],
                ids=[doc_id]
            )
        
        return len(documents)
    
    def add_documents_from_text(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """Add documents directly from text (useful for API)"""
        chunks = self.text_splitter.split_text(text)
        
        safe_chunks = [
                    chunk[:EMBED_MAX_CHARS]
                    for chunk in chunks
                    if chunk and chunk.strip()
]

        documents = []
        for idx, chunk in enumerate(chunks):
            doc_metadata = metadata or {}
            doc_metadata["chunk_id"] = idx
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        existing_count = len(self.vector_store.get()["ids"]) if self.vector_store.get()["ids"] else 0
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        for doc, doc_id in zip(documents, ids):
            self.vector_store.add_documents(
                documents=[doc],
                ids=[doc_id]
    )
        
        return len(documents)
    
    def get_retriever(self, k: int = None):
        """Get retriever for querying"""
        k = k or self.config.RETRIEVAL_K
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """Search for relevant documents"""
        k = k or self.config.RETRIEVAL_K
        retriever = self.get_retriever(k)
        return retriever.invoke(query)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        data = self.vector_store.get()
        total_docs = len(data["ids"]) if data["ids"] else 0
        
        sources = {}
        if data["metadatas"]:
            for metadata in data["metadatas"]:
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": total_docs,
            "sources": sources,
            "collection_name": self.config.COLLECTION_NAME
        }
    
    def clear(self):
        """Clear all documents from the vector store"""
        try:
            self.vector_store.delete_collection()
        except:
            pass
        
        self.vector_store = Chroma(
            collection_name=self.config.COLLECTION_NAME,
            persist_directory=self.config.DB_LOCATION,
            embedding_function=self.embeddings
        )


vector_store = None

def get_vector_store(config: Config = None) -> UniversalVectorStore:
    """Get or create global vector store instance"""
    global vector_store
    if vector_store is None or config is not None:
        vector_store = UniversalVectorStore(config)
    return vector_store
