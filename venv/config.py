# config.py
import os

# --- API Keys ---
# IMPORTANT: Replace these with your actual keys.
# For better security, set them as environment variables.
GEMINI_API_KEY = 'AIzaSyBozQi2V59ZCzUI6smDyDHt1j9sSSkcZbE'
OPENWEATHER_API_KEY = "b9137d6319bf16636be1ef01db243576"
DATAGOV_API_KEY = "your_datagov_api_key_here"

# --- Model Configuration ---
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

# --- File Paths ---
VECTOR_DB_PATH = os.path.join("vector_db", "faiss_index.index")
DOCS_PATH = os.path.join("vector_db", "docs.txt")"

# --- Application Settings ---
# Dictionary of supported languages. Key: language code, Value: display name.
SUPPORTED_LANGUAGES = {
    "ml": "Malayalam",
    "mr": "Marathi",
    "hi": "Hindi",
    "en": "English"
}

# --- Data.gov.in Resource ID ---
MARKET_PRICES_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
