from sentence_transformers import SentenceTransformer,util
from docx import Document
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import re
from keybert import KeyBERT
from faiss import IndexFlatIP
import faiss
import torch

# Defining model parameters
embeddings_model = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
file_path = ""
top_k = 5
model = SentenceTransformer(embeddings_model,device=device)
kw_model = KeyBERT()
custom_stopwords = list(ENGLISH_STOP_WORDS) + ['Atul','Gupta','atul','gupta']


# Function to create vector embeddings for the source document, augment it with synonyms of keywords
# to make sure relevant information is retrieved even if the keyword for it does not exist directly in the document.
# For example, user can ask about school but the document contains education details.
# The embeddings are then stored in a pockle file to avoid re-doing the time consuming process. This can be done because
# the document is mostly static (as is typical for a RAG system).
def create_source_embeddings():    
    # Load document using document library for word documents.
    doc = Document(file_path)
    
    # Reading paras first and then divinf into sentences. Then we store both sentences and paras
    # into text chunk object.
    text_chunks = []
    for para in doc.paragraphs:
        para_text = para.text.strip()
        if not para_text:
            continue
            
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para_text)
        
        # Add paragraph context to each sentence
        for sent in sentences:
            # Store both sentence and its paragraph context. Text chunk is a list of dictionaries where each entry
            # contains a sentence and its corresponding paragraph.
            text_chunks.append({
                'sentence': sent,
                'paragraph_context': para_text
            })

    # Extracting keywords from text_chunks and adding synonyms and relevant keywords to the chunks.
    # This helps the search function search for context dependent words that may not appear in 
    # cosine similarity or other metrics.
    augmented_chunks = []
    for chunk in text_chunks:
        # Extract keywords with context-aware weighting
        keywords = kw_model.extract_keywords(
            chunk['sentence'] + " " + chunk['paragraph_context'],
            keyphrase_ngram_range=(1, 3),
            stop_words=custom_stopwords, # added my name to the stop words otherwise it appends my name to every line
            top_n=3,
            diversity=0.5
        )
        
        # Generating synonyms and appending them to original chunks to create augmented chunks.
        # These augmented chunks form the actual search space in our RAG system.
        augmented_text = chunk['sentence']
        for kw, _ in keywords:
            synonyms = get_contextual_synonyms(kw, chunk['paragraph_context'], model)
            augmented_text += " " + " ".join(synonyms)
        
        augmented_chunks.append(augmented_text.strip())

    # Now we can create a vector embedding for our augmented chunks.
    embeddings = model.encode(augmented_chunks, 
                            batch_size=32,
                            convert_to_tensor=True,
                            show_progress_bar=True)

    # These embeddings are stored in a pickle file to reduce processing times.
    # We can do this because our dataset isn't large and doesn't change very often.
    embeddings = embeddings.cpu() # Moving embeddings to CPU to work with Amazon Linux
    with open('static/document_embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings,augmented_chunks), f)
    
    return augmented_chunks, embeddings

# Function to generate context-aware synonyms using semantic similarity for each keyword 'term'.
def get_contextual_synonyms(term, context, model, top_k=top_k):
    # Encoding available data
    term_embedding = model.encode(term, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)
    
    # Calculating cosine similarity weights.
    # Cosine similarity calculates the cosine of angle between two vectors and uses it to determine if the 
    # vectors are exactly same, completely different, or opposite (1,0,-1)
    similarities = util.pytorch_cos_sim(term_embedding, context_embedding)
    
    # Creating a set of all relevant keywords from the text
    context_words = re.findall(r'\b\w+\b', context.lower())
    unique_words = list(set(context_words))
    
    # Scoring the words by similarity to original term
    word_scores = {}
    for word in unique_words:
        if word == term.lower():
            continue
        word_embedding = model.encode(word, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(term_embedding, word_embedding)
        word_scores[word] = similarity.item()
    
    return [w for w, _ in sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

# Function to create FAISS (FaceBook AI Similarity Search) index and storing it in index file.
# FAISS provides 3 ways of indexing: flat, Inverted file, hierarical navigable small world.
# Flat performs an exhaustive search and that is what we are using.
def create_faiss_index(embeddings):
    # Convert to numpy array first
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings_np)
    
    # Create index
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_np)
    faiss.write_index(index, f"static/faiss_index.index")
    return index


# Function to perform semantic search on the augmented chunks stored as faiss index based on user input.
def semantic_search(query, index, model, augmented_chunks, top_k=top_k):
    # Augmenting query
    query_augmented = augment_query(query, model)
    
    # Encoding the query
    query_embedding = model.encode(query_augmented, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    # FAISS requires 2D array (even for single query)
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)  # Shape: [1, embedding_dim]
    
    # Normalizing user input to match index normalization.
    faiss.normalize_L2(query_embedding)
    
    # Searching the index for top_k nearest neighbours for embedded query.
    # D = Distance, I = Index
    D, I = index.search(query_embedding, top_k)

    return [(augmented_chunks[i], D[0][j]) for j, i in enumerate(I[0])]


# Function to add context aware keywords, synonyms, etc to user input query as well. This makes sure
# our semantic search is able to find relevant information.
def augment_query(query, model, top_k=top_k):
    """Special augmentation for queries"""
    keywords = kw_model.extract_keywords(query, top_n=top_k)
    augmented = query
    for kw, _ in keywords:
        synonyms = get_contextual_synonyms(kw, query, model)
        augmented += " " + " ".join(synonyms)
    return augmented

# Function to load prompt template file and in case the file is missing, return a fallback template.
def load_prompt_template(prompt_file):
    try:
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("prompt file not found")
        return """\n\nHuman: {context_chunks}\nQuestion: {query}\n\nAssistant:"""

# Finction to build an effective llm prompt using user query, context derived from semantic search,
# query's intent as specified, and max_length if specified. Limiting context window saves cost at the expense of 
# accuracy or results.
def build_llm_prompt(query, prompt_file,intent = "context_query", max_length=3000):

    # Reading stored vector embeddings.
    with open('static/document_embeddings.pkl', 'rb') as f:
        embeddings, augmented_chunks  = pickle.load(f)
    embeddings = embeddings.cpu()

    # Reading stored index from index file.
    index = faiss.read_index(f"static/faiss_index.index")

    # Performing semantic search
    search_results = semantic_search(query,index,model,augmented_chunks)

    # Separating the scores from the chunks, keeping only the chunks.
    context_chunks = [chunk for chunk, score in search_results]

    # Truncating chunks to fit model's context window
    truncated_context = []
    total_length = 0
    
    for chunk in context_chunks:
        chunk_length = len(chunk.split())
        if total_length + chunk_length > max_length:
            remaining = max_length - total_length
            if remaining > 50:  # Only add if meaningful space remains
                truncated_context.append(" ".join(chunk.split()[:remaining]))
            break
        truncated_context.append(chunk)
        total_length += chunk_length

    template = load_prompt_template(prompt_file)
    
    # Convert list of chunks to formatted string
    context_str = "\n".join([f"- {chunk}" for chunk in context_chunks])
    print(context_str)

    # Formatting the prompt accordingly and returnin relevant prompt based on query's intent.
    if intent == "greeting":
        print("Intent: greeting")
        return template.format(
            query=query
        )
    else:
        print("Intent: query")
        return template.format(
            context_chunks=context_str,
            query=query
        )

# Setup function to be executed when updating rag document or index
def setup():
    augmented_chunks, embeddings = create_source_embeddings()
    index = create_faiss_index(embeddings)

# Execute this file if you need to change the document and create index and embeddings again.
# This works best if the doc has only paragraphs of text.
if __name__ == "__main__":
    setup()



