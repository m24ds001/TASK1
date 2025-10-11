import os
import gradio as gr
from datetime import datetime
import json
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from bs4 import BeautifulSoup
import threading
import time

class DynamicKnowledgeBase:
    def __init__(self):
        """Initialize the dynamic knowledge base system"""
        print("Initializing Knowledge Base...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise
        
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        self.last_update = None
        
        # Load existing knowledge base if available
        self.load_knowledge_base()
        print(f"✓ Knowledge base initialized with {len(self.documents)} documents")
    
    def add_documents(self, docs: List[str], meta: List[Dict] = None):
        """Add new documents to the knowledge base"""
        if not docs:
            return
        
        try:
            print(f"Adding {len(docs)} documents...")
            embeddings = self.model.encode(docs)
            self.index.add(np.array(embeddings).astype('float32'))
            self.documents.extend(docs)
            
            if meta:
                self.metadata.extend(meta)
            else:
                self.metadata.extend([{"source": "manual", "timestamp": str(datetime.now())} for _ in docs])
            
            self.last_update = datetime.now()
            self.save_knowledge_base()
            print(f"✓ Successfully added {len(docs)} documents")
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if len(self.documents) == 0:
            print("⚠ Knowledge base is empty")
            return []
        
        try:
            print(f"Searching for: '{query}'")
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(k, len(self.documents))
            )
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "score": float(dist)
                    })
            
            print(f"✓ Found {len(results)} results")
            return results
        except Exception as e:
            print(f"✗ Search error: {e}")
            return []
    
    def save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            faiss.write_index(self.index, "knowledge_base.index")
            with open("documents.json", "w") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "last_update": str(self.last_update) if self.last_update else None
                }, f)
            print("✓ Knowledge base saved")
        except Exception as e:
            print(f"✗ Error saving knowledge base: {e}")
    
    def load_knowledge_base(self):
        """Load knowledge base from disk"""
        try:
            if os.path.exists("knowledge_base.index") and os.path.exists("documents.json"):
                self.index = faiss.read_index("knowledge_base.index")
                with open("documents.json", "r") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.metadata = data["metadata"]
                    last_up = data.get("last_update")
                    if last_up:
                        self.last_update = datetime.fromisoformat(last_up)
                print("✓ Loaded existing knowledge base")
        except Exception as e:
            print(f"⚠ Could not load existing knowledge base: {e}")
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_documents": len(self.documents),
            "last_update": str(self.last_update) if self.last_update else "Never",
            "index_size": self.index.ntotal
        }

class DataSourceUpdater:
    def __init__(self, kb: DynamicKnowledgeBase):
        self.kb = kb
        self.sources = []
    
    def add_source(self, source_url: str, source_type: str = "web"):
        """Add a data source for periodic updates"""
        self.sources.append({"url": source_url, "type": source_type})
    
    def fetch_web_content(self, url: str) -> List[str]:
        """Fetch content from a web page"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; KnowledgeBot/1.0)'}
            response = requests.get(url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and split into paragraphs
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_chunks = [chunk for chunk in chunks if chunk and len(chunk) > 50]
            
            return text_chunks[:20]  # Limit to 20 chunks per page
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []
    
    def update_from_sources(self) -> Dict:
        """Update knowledge base from all sources"""
        new_docs = []
        new_meta = []
        
        for source in self.sources:
            if source["type"] == "web":
                content = self.fetch_web_content(source["url"])
                new_docs.extend(content)
                new_meta.extend([{
                    "source": source["url"],
                    "type": "web",
                    "timestamp": str(datetime.now())
                } for _ in content])
        
        if new_docs:
            self.kb.add_documents(new_docs, new_meta)
            return {"status": "success", "new_documents": len(new_docs)}
        return {"status": "no_new_data", "new_documents": 0}

# Initialize global components
print("=" * 50)
print("Starting Dynamic Knowledge Base Chatbot")
print("=" * 50)

kb = DynamicKnowledgeBase()
updater = DataSourceUpdater(kb)

# Add some initial knowledge if empty
if len(kb.documents) == 0:
    print("Adding initial knowledge...")
    initial_docs = [
        "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, learn from it, and make predictions or decisions.",
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. It bridges the gap between human communication and computer understanding.",
        "Deep Learning uses neural networks with multiple layers to progressively extract higher-level features from raw input. It's particularly effective for image and speech recognition.",
        "A chatbot is a software application designed to simulate human conversation through text or voice interactions. Modern chatbots use AI and NLP to understand and respond to user queries.",
        "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model learns to map inputs to outputs based on example input-output pairs.",
        "Unsupervised learning finds hidden patterns in data without pre-existing labels. It's used for clustering, association, and dimensionality reduction tasks.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using a connectionist approach.",
        "Computer vision enables computers to derive meaningful information from digital images and videos. It seeks to automate tasks that human visual systems can do.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties."
    ]
    kb.add_documents(initial_docs)
    print(f"✓ Added {len(initial_docs)} initial documents")

def chatbot_response(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Generate chatbot response using knowledge base"""
    print(f"\n{'='*50}")
    print(f"User query: {message}")
    
    if not message or not message.strip():
        history.append((message, "Please enter a message."))
        return history
    
    try:
        # Search for relevant context
        results = kb.search(message, k=3)
        
        if results:
            # Build response from results
            response_parts = ["**Based on my knowledge base:**\n"]
            
            for i, result in enumerate(results, 1):
                content = result["content"]
                # Truncate long content
                if len(content) > 300:
                    content = content[:300] + "..."
                response_parts.append(f"\n**{i}.** {content}")
            
            response_parts.append(f"\n\n*Retrieved {len(results)} relevant document(s)*")
            response = "\n".join(response_parts)
        else:
            response = "❓ I don't have specific information about that in my knowledge base yet.\n\nYou can:\n- Add new information using the **'Add Knowledge'** tab\n- Configure sources in the **'Data Sources'** tab"
        
        print(f"Response generated: {len(response)} characters")
        history.append((message, response))
        return history
        
    except Exception as e:
        error_msg = f"❌ Error processing your query: {str(e)}\n\nPlease try again or check the logs."
        print(f"✗ Error: {e}")
        history.append((message, error_msg))
        return history

def add_knowledge(text: str) -> str:
    """Add new knowledge to the database"""
    if not text.strip():
        return "⚠️ Please enter some text to add."
    
    try:
        chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip() and len(chunk.strip()) > 10]
        
        if not chunks:
            return "⚠️ Please enter meaningful content (at least 10 characters per line)."
        
        kb.add_documents(chunks)
        stats = kb.get_stats()
        return f"✅ Successfully added {len(chunks)} new document(s)!\n\n📊 Total documents in knowledge base: {stats['total_documents']}"
    except Exception as e:
        return f"❌ Error adding knowledge: {str(e)}"

def add_source(url: str) -> str:
    """Add a new data source"""
    if not url.strip():
        return "⚠️ Please enter a valid URL."
    
    if not url.startswith(('http://', 'https://')):
        return "⚠️ URL must start with http:// or https://"
    
    try:
        updater.add_source(url, "web")
        return f"✅ Added source: {url}\n\nThis source will be checked during updates."
    except Exception as e:
        return f"❌ Error adding source: {str(e)}"

def manual_update() -> str:
    """Manually trigger knowledge base update"""
    try:
        if not updater.sources:
            return "⚠️ No sources configured. Please add sources first!"
        
        result = updater.update_from_sources()
        stats = kb.get_stats()
        
        return f"""✅ Update completed!

📊 Results:
- Status: {result['status']}
- New documents added: {result['new_documents']}
- Total documents: {stats['total_documents']}
- Last update: {stats['last_update']}"""
    except Exception as e:
        return f"❌ Error during update: {str(e)}"

def get_statistics() -> str:
    """Get current knowledge base statistics"""
    try:
        stats = kb.get_stats()
        sources_list = "\n".join([f"  • {s['url']}" for s in updater.sources]) if updater.sources else "  No sources configured yet"
        
        return f"""📊 **Knowledge Base Statistics**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 **Total Documents:** {stats['total_documents']}
🔄 **Last Update:** {stats['last_update']}
💾 **Index Size:** {stats['index_size']}

🔗 **Configured Sources:**
{sources_list}

✅ **Status:** System is operational"""
    except Exception as e:
        return f"❌ Error retrieving statistics: {str(e)}"

# Periodic update function (runs in background)
def periodic_update():
    """Background task for periodic updates"""
    while True:
        time.sleep(3600)  # Update every hour
        try:
            if updater.sources:
                updater.update_from_sources()
                print(f"✓ Automatic update completed at {datetime.now()}")
        except Exception as e:
            print(f"✗ Error in periodic update: {e}")

# Start background update thread
update_thread = threading.Thread(target=periodic_update, daemon=True)
update_thread.start()
print("✓ Background update thread started")

# Create Gradio interface
with gr.Blocks(title="Dynamic Knowledge Base Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Dynamic Knowledge Base Chatbot
    ### A chatbot that automatically expands its knowledge over time
    """)
    
    with gr.Tab("💬 Chat"):
        chatbot_ui = gr.Chatbot(
            value=[],
            height=500,
            label="Conversation",
            show_label=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask me anything... (e.g., 'What is machine learning?')",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        clear_btn = gr.Button("Clear Conversation")
        
        # Event handlers
        submit_btn.click(
            fn=chatbot_response,
            inputs=[msg, chatbot_ui],
            outputs=chatbot_ui
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg
        )
        
        msg.submit(
            fn=chatbot_response,
            inputs=[msg, chatbot_ui],
            outputs=chatbot_ui
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg
        )
        
        clear_btn.click(fn=lambda: [], outputs=chatbot_ui)
    
    with gr.Tab("➕ Add Knowledge"):
        gr.Markdown("### Add new information to the knowledge base")
        gr.Markdown("Enter facts or information (one per line) that you want the chatbot to know about.")
        
        knowledge_input = gr.Textbox(
            label="Enter new knowledge",
            placeholder="Example:\nPython is a high-level programming language.\nIt was created by Guido van Rossum.\nPython emphasizes code readability.",
            lines=10
        )
        add_btn = gr.Button("Add to Knowledge Base", variant="primary")
        add_output = gr.Textbox(label="Result", lines=3)
        
        add_btn.click(add_knowledge, inputs=knowledge_input, outputs=add_output)
    
    with gr.Tab("🔗 Data Sources"):
        gr.Markdown("### Configure automatic data sources")
        gr.Markdown("Add URLs to automatically fetch and update knowledge from web pages.")
        
        source_input = gr.Textbox(
            label="Web Source URL",
            placeholder="https://example.com/article"
        )
        source_btn = gr.Button("Add Source", variant="primary")
        source_output = gr.Textbox(label="Result", lines=2)
        
        gr.Markdown("---")
        gr.Markdown("### Manual Update")
        update_btn = gr.Button("Update from All Sources Now", variant="secondary", size="lg")
        update_output = gr.Textbox(label="Update Result", lines=8)
        
        source_btn.click(add_source, inputs=source_input, outputs=source_output)
        update_btn.click(manual_update, outputs=update_output)
    
    with gr.Tab("📊 Statistics"):
        gr.Markdown("### Knowledge Base Statistics")
        stats_btn = gr.Button("Refresh Statistics", variant="primary")
        stats_output = gr.Textbox(label="Statistics", lines=15)
        
        stats_btn.click(get_statistics, outputs=stats_output)
        demo.load(get_statistics, outputs=stats_output)

print("=" * 50)
print("✓ Gradio interface configured")
print("=" * 50)

# Launch the app
if __name__ == "__main__":
    print("🚀 Launching application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )