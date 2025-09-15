# utils/chatbot_utils.py
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME
from utils.translation_utils import translate_text
from utils.rag_utils import rag_retrieve
import traceback
import io
from PIL import Image

genai.configure(api_key=GEMINI_API_KEY)
# The gemini-1.5-flash model is multimodal, so we can use it for both text and vision
llm_model = genai.GenerativeModel(MODEL_NAME)

def analyze_image_with_gemini(image_bytes: bytes) -> str:
    """
    Uses Gemini's multimodal capability to analyze the provided image.
    """
    print("[CHAT] Step 1.5: Analyzing uploaded image...")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        prompt = "Analyze this image, which is likely from a farm in India. Describe what you see, focusing on crop health, visible pests, diseases, or soil conditions. Be factual, concise, and provide an observation in English."
        
        response = llm_model.generate_content([prompt, image])
        analysis_text = response.text.strip()
        
        print(f"[CHAT] Image analysis result: '{analysis_text[:100]}...'")
        return analysis_text
    except Exception as e:
        print(f"❌ ERROR [CHAT]: Image analysis failed. Reason: {e}")
        return "Error: Could not analyze the uploaded image."


def process_chat_query(query: str, profile: dict, predictions: dict, faiss_index, docs: list[str], language: str, image_bytes: bytes | None = None) -> str:
    """
    Processes a user's chat query, now with optional image analysis.
    """
    print(f"\n--- 🤖 Chatbot Processing Start (Language: {language}) 🤖 ---")
    print(f"[CHAT] Received raw query: '{query}'")

    try:
        # Step 1: Translate user query to English
        print(f"[CHAT] Step 1: Translating query from '{language}' to English...")
        eng_query = translate_text(query, src=language, dest='en')
        if "Translation Error" in eng_query:
            return translate_text("Sorry, I could not understand your query.", src='en', dest=language)

        # Step 1.5: Analyze the image if it exists
        image_analysis = "No image provided."
        if image_bytes:
            image_analysis = analyze_image_with_gemini(image_bytes)

        # Step 2: Retrieve relevant text context using RAG
        print("[CHAT] Step 2: Retrieving text context from knowledge base (RAG)...")
        rag_context = rag_retrieve(eng_query, faiss_index, docs)

        # Step 3: Construct the detailed prompt with all available context
        print("[CHAT] Step 3: Constructing the final prompt for the Gemini model...")
        prompt = f"""
        You are 'Krishi Sakhi', a helpful AI farming assistant for farmers in India.
        Your goal is to provide an accurate and practical solution based on all the information provided.
        Analyze the user's query in the context of their farm profile, AI predictions, image analysis, and knowledge base articles.
        Respond in simple, clear English.

        --- FARMER'S PROFILE ---
        {profile}

        --- AI PREDICTIONS ---
        {predictions}

        --- IMAGE ANALYSIS ---
        Observation from the uploaded image: {image_analysis}

        --- RELEVANT KNOWLEDGE FROM DATABASE---
        {rag_context}

        --- USER'S QUERY ---
        "{eng_query}"

        --- RESPONSE ---
        Provide a helpful and actionable response based on all the information above.
        """

        # Step 4: Call the Gemini model for the final answer
        print("[CHAT] Step 4: Sending final prompt to Gemini model...")
        final_response = llm_model.generate_content(prompt)
        eng_response = final_response.text
        print(f"[CHAT] Received English response from Gemini: '{eng_response[:100]}...'")

        # Step 5: Translate the response back to the user's native language
        print(f"[CHAT] Step 5: Translating response back to '{language}'...")
        native_response = translate_text(eng_response, src='en', dest=language)

        print("--- ✅ Chatbot Processing End ✅ ---\n")
        return native_response

    except Exception as e:
        print(f"❌ FATAL ERROR [CHAT]: An unexpected error occurred. Reason: {e}")
        traceback.print_exc()
        error_message_en = "Sorry, an unexpected error occurred. Please try again."
        return translate_text(error_message_en, src='en', dest=language)