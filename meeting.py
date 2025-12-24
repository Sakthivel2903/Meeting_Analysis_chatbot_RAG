import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

class MeetingNotesRAGSystemEmbeddingsOnly:
    """
    RAG System with ONLY Vector Embeddings (No Structured JSON Storage)
    Uses MongoDB Atlas Vector Search and Sentence Transformers
    """
    
    def __init__(self, groq_api_key: str, mongodb_uri: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize RAG system with embeddings-only support
        """
        # Groq API client for LLM
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        
        # Initialize embedding model (384 dimensions)
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        print(f"✅ Embedding model loaded (dimension: {self.embedding_dimension})")
        
        print(f"🔧 Using Groq Model: {self.model}")
        
        # MongoDB Atlas connection
        try:
            self.mongo_client = MongoClient(mongodb_uri)
            self.db = self.mongo_client['meeting_db']
            
            # Only one collection for embeddings
            self.chunks_col = self.db['document_chunks']
            
            # Test connection
            self.mongo_client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas successfully\n")
            
            # Create indexes
            self.setup_indexes()
            
        except Exception as e:
            print(f"❌ Error connecting to MongoDB Atlas: {e}")
            raise
    
    def setup_indexes(self):
        """Create vector search index"""
        try:
            self.chunks_col.create_index([("created_at", -1)])
            
            print("✅ Database indexes created successfully")
            print("\n⚠️  IMPORTANT: You need to create a Vector Search Index in MongoDB Atlas:")
            print("   1. Go to MongoDB Atlas Dashboard")
            print("   2. Navigate to 'Atlas Search' tab")
            print("   3. Create a Search Index on 'document_chunks' collection")
            print("   4. Use this JSON definition:")
            print("""
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 384,
         "similarity": "cosine"
       }
     ]
   }
            """)
            print("   5. Name it: 'vector_index'\n")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create indexes: {e}\n")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            if not os.path.exists(pdf_path):
                print(f"❌ PDF file not found: {pdf_path}")
                return None
            
            if not pdf_path.lower().endswith('.pdf'):
                print(f"❌ Invalid file format. Please provide a PDF file (.pdf)")
                return None
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"📄 Extracting text from {num_pages} page(s)...")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            print(f"✅ Successfully extracted text from PDF ({len(text)} characters)\n")
            return text
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better embedding
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Sentence Transformers
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def check_file_exists(self, pdf_filename: str) -> bool:
        """
        Check if PDF file already exists in database
        """
        try:
            existing = self.chunks_col.find_one({"source_file": pdf_filename})
            return existing is not None
        except Exception as e:
            print(f"❌ Error checking file existence: {e}")
            return False
    
    def get_file_info(self, pdf_filename: str) -> Dict:
        """
        Get information about an existing file
        """
        try:
            count = self.chunks_col.count_documents({"source_file": pdf_filename})
            first_doc = self.chunks_col.find_one({"source_file": pdf_filename})
            
            return {
                "exists": True,
                "chunk_count": count,
                "created_at": first_doc.get('created_at') if first_doc else None
            }
        except Exception as e:
            print(f"❌ Error getting file info: {e}")
            return {"exists": False}
    
    def store_embeddings(self, meeting_notes: str, pdf_filename: str = "meeting_notes.pdf") -> bool:
        """
        Store ONLY embeddings (no structured JSON data)
        """
        try:
            print("🔄 Creating embeddings for vector search...")
            
            # Chunk the meeting notes
            chunks = self.chunk_text(meeting_notes, chunk_size=500, overlap=50)
            print(f"📦 Created {len(chunks)} text chunks")
            
            # Generate embeddings and store
            embeddings_stored = 0
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.generate_embedding(chunk)
                
                # Store ONLY: text, embedding, and minimal metadata
                chunk_doc = {
                    "chunk_index": idx,
                    "text": chunk,
                    "embedding": embedding,
                    "source_file": pdf_filename,
                    "created_at": datetime.utcnow()
                }
                
                self.chunks_col.insert_one(chunk_doc)
                embeddings_stored += 1
            
            print(f"✅ Stored {embeddings_stored} embeddings in database")
            print(f"✅ Embeddings stored successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error storing embeddings: {e}")
            return False
    
    def list_all_files(self) -> List[str]:
        """
        List all unique PDF files stored in the database
        """
        try:
            files = self.chunks_col.distinct("source_file")
            return files
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 100,
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "source_file": 1,
                        "chunk_index": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.chunks_col.aggregate(pipeline))
            return results
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            print("⚠️  Make sure you've created the vector search index in MongoDB Atlas!")
            return []
    
    def answer_question(self, user_query: str) -> str:
        """
        Answer user questions using vector similarity search + RAG
        """
        print("🔄 Converting question to embedding...")
        
        # Step 1: Perform vector similarity search
        print("🔍 Searching for relevant context...")
        search_results = self.vector_search(user_query, top_k=5)
        
        if not search_results:
            return "❌ No relevant information found. Make sure:\n1. You've processed a PDF file\n2. Vector search index is created in MongoDB Atlas"
        
        # Step 2: Build context from search results
        context_parts = []
        context_parts.append("RELEVANT INFORMATION FROM DOCUMENTS:\n")
        
        for idx, result in enumerate(search_results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')
            source = result.get('source_file', 'Unknown')
            chunk_idx = result.get('chunk_index', 0)
            
            context_parts.append(f"\n[Result {idx}] (Relevance Score: {score:.4f})")
            context_parts.append(f"Source: {source} (Chunk #{chunk_idx})")
            context_parts.append(f"Content: {text}")
            context_parts.append("-" * 80)
        
        context = "\n".join(context_parts)
        
        # Step 3: Generate answer using LLM with retrieved context
        prompt = f"""You are a professional assistant analyzing meeting notes and documents.

{context}

USER QUESTION: {user_query}

Based on the relevant information above, provide a clear and accurate answer.
- Focus on the most relevant results (higher relevance scores)
- If information is not in the context, say so clearly
- Be concise and professional
- Use bullet points where appropriate"""

        try:
            print("💭 Generating answer with LLM...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant. Answer based only on the provided context from vector similarity search."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            answer = chat_completion.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            print("✅ MongoDB connection closed")


def main():
    """Example usage of the RAG system with embeddings only"""
    
    # Get credentials from environment variables
    groq_api_key = os.environ.get("GROQ_API_KEY")
    mongodb_uri = os.environ.get("MONGODB_URI")
    
    if not groq_api_key:
        print("⚠️  Please set your GROQ_API_KEY environment variable")
        return
    
    if not mongodb_uri:
        print("⚠️  Please set your MONGODB_URI environment variable")
        return
    
    try:
        print("="*100)
        print("🚀 MEETING NOTES RAG SYSTEM - EMBEDDINGS ONLY")
        print("="*100 + "\n")
        
        rag_system = MeetingNotesRAGSystemEmbeddingsOnly(
            groq_api_key=groq_api_key,
            mongodb_uri=mongodb_uri,
            model="llama-3.3-70b-versatile"
        )
        
        # Step 1: Upload PDF
        print("="*100)
        print("STEP 1: UPLOAD DOCUMENT PDF")
        print("="*100)
        pdf_path = input("\n📄 Enter PDF file path: ").strip()
        
        if not pdf_path:
            print("❌ No file path provided!")
            return
        
        pdf_filename = os.path.basename(pdf_path)
        
        # Step 2: Check if file already exists
        print("\n🔍 Checking if file already exists in database...")
        file_exists = rag_system.check_file_exists(pdf_filename)
        
        if file_exists:
            print(f"✅ File '{pdf_filename}' already exists in database!")
            
            # Get file info
            file_info = rag_system.get_file_info(pdf_filename)
            print(f"📊 Existing file info:")
            print(f"   - Chunks stored: {file_info.get('chunk_count', 0)}")
            print(f"   - Added on: {file_info.get('created_at', 'N/A')}")
            
            # Ask user what to do
            print("\n❓ What would you like to do?")
            print("   1. Use existing file (skip processing)")
            print("   2. Re-process and replace existing file")
            print("   3. Cancel")
            
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print(f"\n✅ Using existing file '{pdf_filename}' from database")
                print("⏭️  Skipping to question answering...\n")
            elif choice == "2":
                print(f"\n🗑️  Deleting old embeddings for '{pdf_filename}'...")
                deleted_count = rag_system.chunks_col.delete_many({"source_file": pdf_filename}).deleted_count
                print(f"✅ Deleted {deleted_count} old embeddings")
                
                # Extract and store new
                print("\n📄 Processing PDF file...")
                document_text = rag_system.extract_text_from_pdf(pdf_path)
                
                if not document_text:
                    print("❌ Failed to extract text from PDF!")
                    return
                
                print("="*100)
                print("STEP 2: CREATING AND STORING NEW EMBEDDINGS")
                print("="*100 + "\n")
                
                success = rag_system.store_embeddings(document_text, pdf_filename)
                
                if not success:
                    print("❌ Failed to store embeddings!")
                    return
            else:
                print("\n❌ Operation cancelled!")
                return
        else:
            print(f"📝 File '{pdf_filename}' is NEW - will process and store")
            
            # Extract text
            print("\n📄 Processing PDF file...")
            document_text = rag_system.extract_text_from_pdf(pdf_path)
            
            if not document_text:
                print("❌ Failed to extract text from PDF!")
                return
            
            # Store embeddings
            print("="*100)
            print("STEP 2: CREATING AND STORING EMBEDDINGS")
            print("="*100 + "\n")
            
            success = rag_system.store_embeddings(document_text, pdf_filename)
            
            if not success:
                print("❌ Failed to store embeddings!")
                return
        
        # Step 4: Ask questions using vector search
        print("="*100)
        print("STEP 3: ASK QUESTIONS (POWERED BY VECTOR SIMILARITY SEARCH)")
        print("="*100)
        print("\n💬 Your questions will be converted to embeddings and matched with relevant content.")
        print("📁 Commands:")
        print("   - Type 'list' to see all files in database")
        print("   - Type 'exit' to quit\n")
        
        while True:
            user_question = input("❓ Your question: ").strip()
            
            if user_question.lower() == 'exit':
                print("\n👋 Thank you for using the RAG System!")
                break
            elif user_question.lower() == 'list':
                print("\n📁 Files in database:")
                files = rag_system.list_all_files()
                if files:
                    for idx, file in enumerate(files, 1):
                        info = rag_system.get_file_info(file)
                        print(f"   {idx}. {file} ({info.get('chunk_count', 0)} chunks)")
                else:
                    print("   No files found in database")
                print()
            elif user_question:
                print()
                answer = rag_system.answer_question(user_question)
                print(f"\n💡 Answer:\n{answer}\n")
                print("-"*100 + "\n")
            else:
                print("⚠️  Please enter a question\n")
        
        rag_system.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
