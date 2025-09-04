# Disable ChromaDB telemetry BEFORE any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
import openai
from typing import List, Dict, Any
from utils.pdf_processor import PDFProcessor

class VectorStore:
    def __init__(self, persist_directory: str = "data/vectorstore", openai_api_key: str = None):
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with telemetry completely disabled
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
        except Exception as e:
            print(f"Warning: ChromaDB initialization issue: {e}")
            # Fallback to basic client
            self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="flight_regulations_v2",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 50
            }
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
        """Add documents to the vector store with enhanced processing"""
        if not documents:
            return
        
        # Extract texts and prepare metadata
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Create more descriptive IDs
        ids = [f"{doc['metadata']['source']}_{doc['metadata']['chunk_id']}_{doc['metadata']['content_hash']}" 
               for doc in documents]
        
        # Get embeddings in batches for better performance
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
            
            # Print summary of content types
            content_types = {}
            for doc in documents:
                content_type = doc['metadata'].get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            print(f"Content type distribution: {content_types}")
    
    def reset_collection(self):
        """Reset the collection to start fresh"""
        try:
            self.client.delete_collection("flight_regulations_v2")
            self.collection = self.client.get_or_create_collection(
                name="flight_regulations_v2",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:search_ef": 50
                }
            )
            print("Collection reset successfully")
        except Exception as e:
            print(f"Error resetting collection: {e}")
    
    def initialize_from_pdfs(self, pdf_folder: str = "data/regulations", force_reload: bool = False):
        """Initialize vector store from PDF files with enhanced chunking"""
        processor = PDFProcessor(pdf_folder)
        
        # Check if collection is already populated
        if self.collection.count() > 0 and not force_reload:
            print("Vector store already initialized. Use force_reload=True to reinitialize.")
            return
        
        if force_reload:
            print("Force reloading vector store...")
            self.reset_collection()
        
        print("Processing PDFs with enhanced chunking strategy...")
        processed_docs = processor.process_all_pdfs()
        
        # Flatten all chunks into a single list
        all_chunks = []
        for filename, chunks in processed_docs.items():
            all_chunks.extend(chunks)
        
        if all_chunks:
            self.add_documents(all_chunks)
            print(f"Vector store initialized with {len(all_chunks)} chunks from {len(processed_docs)} documents")
            
            # Print statistics
            self._print_collection_stats()
        else:
            print("No PDFs found to process")
    
    def _print_collection_stats(self):
        """Print statistics about the collection"""
        try:
            count = self.collection.count()
            print(f"\n📊 Collection Statistics:")
            print(f"   Total chunks: {count}")
            
            # Get sample to analyze content types
            sample = self.collection.get(limit=min(100, count))
            content_types = {}
            regulation_types = {}
            
            for metadata in sample['metadatas']:
                content_type = metadata.get('content_type', 'unknown')
                regulation_type = metadata.get('regulation_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                regulation_types[regulation_type] = regulation_types.get(regulation_type, 0) + 1
            
            print(f"   Content types: {content_types}")
            print(f"   Regulation types: {regulation_types}")
            
        except Exception as e:
            print(f"Error printing collection stats: {e}")
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Dict = None, 
               content_type_filter: str = None, regulation_type_filter: str = None,
               boost_compensation: bool = False) -> List[Dict]:
        """Enhanced search with filtering and ranking options"""
        try:
            query_embedding = self.get_embeddings([query])[0]
            
            # Build search parameters
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results * 2, 20)  # Get more results for re-ranking
            }
            
            # Apply filters
            where_conditions = {}
            if filter_metadata:
                where_conditions.update(filter_metadata)
            if content_type_filter:
                where_conditions["content_type"] = content_type_filter
            if regulation_type_filter:
                where_conditions["regulation_type"] = regulation_type_filter
            
            if where_conditions:
                search_kwargs["where"] = where_conditions
            
            results = self.collection.query(**search_kwargs)
            
            # Format and rank results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i],
                    'relevance_score': self._calculate_relevance_score(
                        query, results['documents'][0][i], 
                        results['metadatas'][0][i], boost_compensation
                    )
                }
                formatted_results.append(result)
            
            # Sort by relevance score and return top results
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return formatted_results[:n_results]
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def _calculate_relevance_score(self, query: str, content: str, metadata: Dict, 
                                 boost_compensation: bool = False) -> float:
        """Calculate enhanced relevance score combining semantic similarity and metadata"""
        # Base score from vector similarity (invert distance)
        base_score = 1.0 - metadata.get('distance', 1.0)
        
        # Boost for compensation-related content if requested
        compensation_boost = 0.0
        if boost_compensation and metadata.get('has_compensation_info', False):
            compensation_boost = 0.2
        
        # Boost for content type relevance
        content_type_boost = 0.0
        content_type = metadata.get('content_type', 'general')
        if content_type in ['compensation', 'delay_provision']:
            content_type_boost = 0.1
        elif content_type in ['article', 'section']:
            content_type_boost = 0.05
        
        # Boost for key terms matching
        key_terms_boost = 0.0
        query_terms = set(query.lower().split())
        chunk_terms_str = metadata.get('key_terms', '')
        if chunk_terms_str:
            chunk_terms = set(chunk_terms_str.split('|'))
            if query_terms & chunk_terms:
                key_terms_boost = 0.1 * len(query_terms & chunk_terms) / len(query_terms)
        
        # Combine scores
        total_score = base_score + compensation_boost + content_type_boost + key_terms_boost
        return min(total_score, 1.0)  # Cap at 1.0
    
    def search_by_content_type(self, query: str, content_type: str, n_results: int = 3) -> List[Dict]:
        """Search specifically for a content type"""
        return self.search(query, n_results, content_type_filter=content_type)
    
    def search_compensation_info(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search specifically for compensation-related information"""
        return self.search(query, n_results, boost_compensation=True)
    
    def get_related_chunks(self, chunk_id: str, n_results: int = 3) -> List[Dict]:
        """Get chunks related to a specific chunk"""
        try:
            # Get the chunk content
            chunk_data = self.collection.get(ids=[chunk_id])
            if not chunk_data['documents']:
                return []
            
            chunk_content = chunk_data['documents'][0]
            chunk_metadata = chunk_data['metadatas'][0]
            
            # Search for similar content
            return self.search(
                chunk_content, 
                n_results=n_results + 1,  # +1 because original chunk will be included
                regulation_type_filter=chunk_metadata.get('regulation_type')
            )
        except Exception as e:
            print(f"Error getting related chunks: {e}")
            return []