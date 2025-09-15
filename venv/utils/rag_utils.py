# utils/rag_utils.py
import faiss
import numpy as np
import google.generativeai as genai
from config import EMBEDDING_MODEL, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def embed_texts(texts: list[str]) -> np.ndarray:
    """Generates embeddings for a list of texts."""
    print(f"[RAG] Attempting to embed {len(texts)} text snippet(s)...")
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        print("[RAG] Embedding successful.")
        return np.array(result['embedding'])
    except Exception as e:
        print(f"❌ ERROR [RAG]: Failed to generate embeddings. Reason: {e}")
        raise

def load_faiss_index(index_path: str, docs_path: str):
    """Loads the FAISS index and documents with detailed error handling."""
    print("[LOAD] Attempting to load FAISS index and documents...")
    try:
        index = faiss.read_index(index_path)
        print(f"[LOAD] FAISS index loaded from '{index_path}'.")
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs = f.read().split('\n---\n')
        print(f"[LOAD] Documents loaded from '{docs_path}'.")
        return index, docs
    except FileNotFoundError as e:
        print(f"❌ ERROR [LOAD]: Could not find a required file. Reason: {e}")
        print("Please ensure you have run 'build_vector_db.py' successfully.")
        return None, None
    except Exception as e:
        print(f"❌ ERROR [LOAD]: An unexpected error occurred while loading resources. Reason: {e}")
        return None, None

def rag_retrieve(query: str, index, docs: list[str], k: int = 3) -> str:
    """Retrieves relevant documents from the vector DB."""
    print(f"[RAG] Retrieving context for query: '{query[:50]}...'")
    try:
        query_embedding = embed_texts([query])
        _, indices = index.search(query_embedding, k)
        relevant_docs = [docs[i] for i in indices[0] if i < len(docs)]
        context = "\n---\n".join(relevant_docs)
        print(f"[RAG] Retrieved {len(relevant_docs)} context snippets.")
        return context
    except Exception as e:
        print(f"❌ ERROR [RAG]: Failed during context retrieval. Reason: {e}")
        return "Error: Could not retrieve relevant context."