import chromadb
from chromadb.config import Settings
import openai
from typing import List, Dict, Any
import os
from utils.pdf_processor import PDFProcessor

class VectorStore:
    def __init__(self, persist_directory: str = "data/vectorstore", openai_api_key: str = None):
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="flight_regulations",
            metadata={"hnsw:space": "cosine"}
        )
        
        openai.api_key = self.openai_api_key
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI's text-embedding-3-small model"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        if not documents:
            return
        
        # Extract texts and prepare metadata
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"{doc['metadata']['source']}_{doc['metadata']['chunk_id']}" for doc in documents]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        if embeddings:
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents to vector store")
    
    def initialize_from_pdfs(self, pdf_folder: str = "data/regulations"):
        """Initialize vector store from PDF files"""
        processor = PDFProcessor(pdf_folder)
        
        # Check if collection is already populated
        if self.collection.count() > 0:
            print("Vector store already initialized")
            return
        
        print("Processing PDFs and initializing vector store...")
        processed_docs = processor.process_all_pdfs()
        
        # Flatten all chunks into a single list
        all_chunks = []
        for filename, chunks in processed_docs.items():
            all_chunks.extend(chunks)
        
        if all_chunks:
            self.add_documents(all_chunks)
            print(f"Vector store initialized with {len(all_chunks)} chunks from {len(processed_docs)} documents")
        else:
            print("No PDFs found to process")
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        """Search for relevant documents"""
        try:
            query_embedding = self.get_embeddings([query])[0]
            
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            if filter_metadata:
                search_kwargs["where"] = filter_metadata
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []