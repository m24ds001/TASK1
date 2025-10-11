Problem Statement
The challenge is to build a chatbot that can learn and adapt by automatically expanding its knowledge base over time. Unlike static chatbots that rely on a fixed dataset, this system must be dynamic. The goal is to create a mechanism that periodically updates the chatbot's internal information with new content from specified sources, such as web pages, so it can provide relevant and current responses.

Dataset
The project does not use a pre-defined, static dataset. Instead, the "dataset" is dynamically created and expanded during runtime.

Initial Knowledge: The system is pre-populated with a small set of foundational documents related to AI, machine learning, and NLP. These serve as a starting point for the knowledge base.

Dynamic Expansion: The knowledge base is continuously expanded with new data fetched from user-defined web sources. The system extracts meaningful text chunks from these sources to add to its knowledge.

Data Format: The documents are stored as plain text strings, along with metadata such as the source URL and timestamp. These documents are then converted into high-dimensional vectors for a similarity search.

Methodology
The system's methodology is based on Retrieval-Augmented Generation (RAG) principles, integrating a vector database with a conversational interface.

Vectorization: Textual documents are transformed into numerical vector embeddings using a pre-trained SentenceTransformer model (all-MiniLM-L6-v2). This process captures the semantic meaning of the text.

Vector Storage: The resulting embeddings are stored in a faiss vector database. faiss is optimized for efficient similarity search, which is crucial for quickly finding relevant information. An alternative, lightweight approach using TfidfVectorizer and cosine_similarity is also provided.

Dynamic Updates:

A DataSourceUpdater class is responsible for fetching new information from specified URLs.

The BeautifulSoup4 library is used to scrape and parse text from HTML content.

A background thread is initiated to perform these updates automatically and periodically (e.g., every hour).

Information Retrieval: When a user submits a query, it is first converted into a vector embedding. The faiss index is then searched to retrieve the most semantically similar documents from the knowledge base.

Response Generation: The retrieved documents are used as context to formulate a relevant and informative response for the user, citing the source of the information.

Results
The implemented system successfully demonstrates the ability of a chatbot to dynamically expand its knowledge base and provide more comprehensive responses over time.

Knowledge Growth: The knowledge base size, tracked by the total_documents statistic, increases with each manual addition or periodic update from a data source. This confirms the system's ability to incorporate new information.

Enhanced Responses: The chatbot is able to answer queries based on both its initial knowledge and the new information it has learned. Queries related to newly added sources yield relevant results from the knowledge base, showing that the system has successfully integrated the new data.

System Stability: The use of a background thread for updates ensures that the learning process does not interrupt the chatbot's main conversational function. The system status is consistently reported as operational.

Reproducibility: The project's dependencies are clearly documented in requirements.txt, making it easy to reproduce the environment and verify the results.

Alternative Implementation: The inclusion of app_lightweight.py provides a more resource-efficient version for deployments with hardware constraints. This version removes duplicate documents after each update to maintain a clean knowledge base.
