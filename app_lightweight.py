import gradio as gr
from datetime import datetime
import json
import os
from typing import List, Dict
import pickle
import requests
from bs4 import BeautifulSoup
import threading
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LightweightKnowledgeBase:
    """Lightweight knowledge base using TF-IDF instead of transformers"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.metadata = []
        self.document_vectors = None
        self.last_update = None
        self.load_knowledge_base()
    
    def add_documents(self, docs: List[str], meta: List[Dict] = None):
        """Add new documents to the knowledge base"""
        if not docs:
            return
        
        self.documents.extend(docs)
        
        if meta:
            self.metadata.extend(meta)
        else:
            self.metadata.extend([{
                "source": "manual",
                "timestamp": str(datetime.now())
            } for _ in docs])
        
        # Refit vectorizer with all documents
        if len(self.documents) > 0:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        self.last_update = datetime.now()
        self.save_knowledge_base()
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents using cosine similarity"""
        if len(self.documents) == 0 or self.document_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "score": float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "last_update": str(self.last_update) if self.last_update else None
            }
            
            with open("kb_data.json", "w") as f:
                json.dump(data, f)
            
            with open("vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            if self.document_vectors is not None:
                with open("vectors.pkl", "wb") as f:
                    pickle.dump(self.document_vectors, f)
                    
        except Exception as e:
            print(f"Error saving: {e}")
    
    def load_knowledge_base(self):
        """Load knowledge base from disk"""
        try:
            if os.path.exists("kb_data.json"):
                with open("kb_data.json", "r") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", [])
                    last_up = data.get("last_update")
                    if last_up:
                        self.last_update = datetime.fromisoformat(last_up)
            
            if os.path.exists("vectorizer.pkl"):
                with open("vectorizer.pkl", "rb") as f:
                    self.vectorizer = pickle.load(f)
            
            if os.path.exists("vectors.pkl") and len(self.documents) > 0:
                with open("vectors.pkl", "rb") as f:
                    self.document_vectors = pickle.load(f)
                    
        except Exception as e:
            print(f"Error loading: {e}")
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_documents": len(self.documents),
            "last_update": str(self.last_update) if self.last_update else "Never",
            "vocabulary_size": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
        }
    
    def remove_duplicates(self):
        """Remove duplicate documents"""
        seen = set()
        unique_docs = []
        unique_meta = []
        
        for doc, meta in zip(self.documents, self.metadata):
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)
                unique_meta.append(meta)
        
        self.documents = unique_docs
        self.metadata = unique_meta
        
        if len(self.documents) > 0:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        self.save_knowledge_base()

class DataSourceManager:
    """Manages data sources and updates"""
    
    def __init__(self, kb: LightweightKnowledgeBase):
        self.kb = kb
        self.sources = []
    
    def add_source(self, url: str):
        """Add a data source"""
        if url not in [s["url"] for s in self.sources]:
            self.sources.append({
                "url": url,
                "last_fetched": None,
                "status": "pending"
            })
    
    def fetch_web_content(self, url: str, max_chunks: int = 15) -> List[str]:
        """Fetch and parse web content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; KnowledgeBot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all(['p', 'article', 'section'])
            chunks = []
            
            for para in paragraphs:
                text = para.get_text().strip()
                if len(text) > 100:  # Minimum length
                    chunks.append(text)
                    if len(chunks) >= max_chunks:
                        break
            
            return chunks
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []
    
    def update_from_sources(self) -> Dict:
        """Update knowledge base from all sources"""
        total_new = 0
        successful = 0
        
        for source in self.sources:
            try:
                content = self.fetch_web_content(source["url"])
                
                if content:
                    metadata = [{
                        "source": source["url"],
                        "type": "web",
                        "timestamp": str(datetime.now())
                    } for _ in content]
                    
                    self.kb.add_documents(content, metadata)
                    total_new += len(content)
                    successful += 1
                    source["last_fetched"] = str(datetime.now())
                    source["status"] = "success"
                else:
                    source["status"] = "no_content"
                    
            except Exception as e:
                source["status"] = f"error: {str(e)}"
        
        # Remove duplicates after updating
        self.kb.remove_duplicates()
        
        return {
            "new_documents": total_new,
            "sources_updated": successful,
            "total_sources": len(self.sources)
        }

# Initialize system
kb = LightweightKnowledgeBase()
manager = DataSourceManager(kb)

# Add initial knowledge
if len(kb.documents) == 0:
    initial_knowledge = [
        "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
        "Machine Learning is a subset of AI that enables systems to automatically learn and improve from experience.",
        "Natural Language Processing helps computers understand, interpret and generate human language.",
        "Deep Learning uses neural networks with multiple layers to analyze various factors of data.",
        "A chatbot is an AI-powered software that can simulate conversations with users.",
        "Python is a popular programming language widely used in AI and data science.",
        "Vector databases store data as high-dimensional vectors for similarity search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "RAG (Retrieval Augmented Generation) combines retrieval and generation for better AI responses.",
        "Fine-tuning adapts pre-trained models to specific tasks or domains."
    ]
    kb.add_documents(initial_knowledge)

def chat_response(message: str, history: List) -> str:
    """Generate response using knowledge base"""
    if not message.strip():
        return "Please enter a message."
    
    results = kb.search(message, k=3)
    
    if results:
        context_parts = []
        for i, r in enumerate(results, 1):
            score_pct = int(r["score"] * 100)
            context_parts.append(f"{i}. {r['content']} (relevance: {score_pct}%)")
        
        context = "\n\n".join(context_parts)
        response = f"📚 **Relevant Information:**\n\n{context}"
    else:
        response = "❓ I don't have specific information about that yet. Try adding relevant knowledge or sources!"
    
    return response

def add_knowledge_handler(text: str) -> str:
    """Handle adding new knowledge"""
    if not text.strip():
        return "⚠️ Please enter some text."
    
    chunks = [line.strip() for line in text.split("\n") if line.strip() and len(line.strip()) > 20]
    
    if not chunks:
        return "⚠️ Please enter meaningful content (at least 20 characters per line)."
    
    kb.add_documents(chunks)
    return f"✅ Added {len(chunks)} new document(s)!\n\nTotal documents: {len(kb.documents)}"

def add_source_handler(url: str) -> str:
    """Handle adding new source"""
    if not url.strip():
        return "⚠️ Please enter a URL."
    
    if not url.startswith(('http://', 'https://')):
        return "⚠️ URL must start with http:// or https://"
    
    manager.add_source(url)
    return f"✅ Added source: {url}\n\nThis will be checked during updates."

def update_now_handler() -> str:
    """Handle manual update trigger"""
    if not manager.sources:
        return "⚠️ No sources configured. Add sources first!"
    
    result = manager.update_from_sources()
    stats = kb.get_stats()
    
    return f"""✅ **Update Complete**

📊 Results:
- New documents: {result['new_documents']}
- Sources updated: {result['sources_updated']}/{result['total_sources']}
- Total documents: {stats['total_documents']}
- Last update: {stats['