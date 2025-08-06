## <!-- Talk to my Assistant (not live anymore): https://tinyurl.com/mr4xrty -->

<img width="745" alt="image" src="https://github.com/user-attachments/assets/2cfa841c-8f40-4db7-b7a7-c650da9aa0f4" />

# RAG AI Personal Assistant

RAG AI Personal Assistant is a Retrieval-Augmented Generation (RAG) application designed to help hiring teams and recruiters gain deep insights about a job candidate.
The system leverages vector embeddings and NLP techniques like semantic search to learn from a candidate’s personal documents, and then uses a Large Language Model (LLM) to generate natural language responses to queries about the candidate’s profile.
Users can then ask questions about the candidate, and the system provides context-aware, detailed responses based on the provided document.

## Key Features

- **Vector Embeddings & Semantic Search:** Extracts meaningful features from candidate documents for robust search and retrieval.
- **LLM-powered Query Response:** Utilizes a large language model (Anthropic Claude-v2) to generate human-like, contextually relevant answers.
- **RAG Framework:** Combines retrieval of candidate-specific data with generation for improved answer quality.
- **Secure Deployment:** Hosted on AWS Elastic Beanstalk with SSL configuration for secure, reliable access.

## Architecture

The project architecture is centered on two core components:

1. **Data Processing & Embedding Generation:**
   - Ingests and preprocesses candidate documents.
   - Parses the document by sentence and by paragraphs. Augments the text with contextual keywords using keybert.
   - Generates vector embeddings for semantic search (using all-MiniLM-L6-v2 with 384 dimensions).
   - Creates a FAISS (Facebook AI Similarity Search) index of the embeddings.
   - Reads user query and augments it similarly, then uses cosine similarity to gather context from the document.

2. **Query Processing & Response Generation:**
   - Uses the vector search results to retrieve relevant candidate information.
   - Feeds the retrieved information into an LLM to generate responses.

The entire application is deployed on AWS Elastic Beanstalk as a Flask application, served by Nginx and gunicorn.

![image](https://github.com/user-attachments/assets/9b9f60d7-719b-4e5c-9d08-b0d4f2b3d4a7)

