# app.py
import streamlit as st
import os
import faiss
import numpy as np
import google.generativeai as genai

# Corrected Imports
import config
from utils.rag_utils import load_faiss_index, embed_texts
from utils.ml_utils import compute_predictions
from utils.chatbot_utils import process_chat_query
from utils.api_utils import fetch_weather, fetch_gov_info # Make sure this import exists

# Define the paths
VECTOR_DB_PATH = os.path.join("vector_db", "faiss_index.index")
DOCS_PATH = os.path.join("vector_db", "docs.txt")


# --- Load Vector DB ---
# This is loaded once and cached for performance
@st.cache_resource
def load_resources():
    """
    Checks for the existence of the vector DB files and loads them.
    Returns the FAISS index and the documents, or None if not found.
    """
    print("[APP] Attempting to load resources (FAISS index and docs)...")
    
    # Safety check to ensure files exist before trying to load them
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(DOCS_PATH):
        print(f"❌ [APP] Error: Vector DB files not found at expected paths.")
        return None, None
    
    # Load the files using the helper function
    index, docs = load_faiss_index(VECTOR_DB_PATH, DOCS_PATH)
    print("✅ [APP] Resources loaded successfully.")
    return index, docs

# --- Execute the loading function ---
# This line runs the function and stores the result in variables for the app to use.
faiss_index, docs = load_resources()

# --- Main Application Logic Starts Below ---
# (e.g., your st.title(), main(), etc.)

# --- Page Configuration ---
st.set_page_config(page_title="Krishi Sakhi", page_icon="🌱", layout="wide")

# --- State Management ---
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
# Chat history now stores dictionaries for flexibility with images
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# --- Load Resources ---
@st.cache_resource
def load_resources():
    print("[APP] Loading resources (FAISS index and docs)...")
    index, docs = load_faiss_index(VECTOR_DB_PATH, DOCS_PATH)
    return index, docs

faiss_index, docs = load_resources()

# --- Main Application (No Changes Here) ---
def main():
    if not GEMINI_API_KEY or "your_gemini_api_key" in GEMINI_API_KEY:
        st.error("🚨 Gemini API key is not configured! Please set it in config.py.")
        return
    if not faiss_index:
        st.error("🚨 Vector DB not found! Please run `build_vector_db.py` first.")
        return

    st.title("🌱 Krishi Sakhi POC")
    st.caption("Your AI Farming Companion in your pocket, anywhere, anytime.")

    tab1, tab2, tab3 = st.tabs(["👤 Farm Profile", "📊 Dashboard", "💬 AI Chatbot"])

    with tab1:
        profiling_section()
    with tab2:
        dashboard_section()
    with tab3:
        chatbot_section()

# --- Profiling and Dashboard Sections (No Changes Here) ---
def profiling_section():
    st.header("👤 Farmer & Farm Profiling")
    st.write("Enter your details to get personalized advice.")
    with st.form("profiling_form"):
        profile = st.session_state.profile or {}
        c1, c2 = st.columns(2)
        with c1:
            village = st.text_input("Village", value=profile.get("village", "Pune"))
            crop = st.selectbox("Primary Crop", ["Paddy", "Banana", "Brinjal", "Coconut", "Tomato", "Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Jute", "Oilseeds", "Pulses", "Fruits", "Vegetables", "Others"], index=0)
            land_size = st.number_input("Land Size (acres)", min_value=0.1, value=profile.get("land_size", 2.0))
        with c2:
            soil = st.selectbox("Soil Type", ["Sandy", "Clay", "Laterite", "Black", "Red"], index=1)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=profile.get("ph", 6.5))
            language = st.selectbox("Preferred Language", options=list(SUPPORTED_LANGUAGES.keys()), format_func=lambda code: SUPPORTED_LANGUAGES[code], index=1)
        
        if st.form_submit_button("Save Profile"):
            st.session_state.profile = {"village": village, "crop": crop, "soil": soil, "land_size": land_size, "ph": ph, "language": language}
            st.session_state.predictions = compute_predictions(st.session_state.profile)
            st.success("✅ Profile saved successfully!")

def dashboard_section():
    st.header("📊 Personalized Dashboard")
    if not st.session_state.profile:
        st.info("Please complete your profile in the 'Farm Profile' tab.")
        return
    profile, predictions = st.session_state.profile, st.session_state.predictions
    weather, gov_info = fetch_weather(profile['village']), fetch_gov_info(profile['crop'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌦️ Weather")
        if 'error' in weather: st.warning(weather['error'])
        else:
            st.metric("Temperature", f"{weather['temperature']} °C")
            st.metric("Condition", weather['weather'])
    with col2:
        st.subheader("📈 ML Predictions")
        st.metric("Estimated Yield", f"{predictions.get('yield', 'N/A')} kg/acre")
        st.metric("Pest Risk", f"{predictions.get('pest_risk', 0) * 100:.0f}%")
    with col3:
        st.subheader("🏦 Market & Schemes")
        if 'error' in gov_info: st.warning(gov_info['error'])
        else: st.metric(f"Price for {profile['crop']}", f"₹ {gov_info.get('market_price', 'N/A')} /Quintal")

# --- Chatbot Section (UPDATED) ---
def chatbot_section():
    st.header("💬 Chat with Krishi Sakhi")
    
    if not st.session_state.profile:
        st.info("Please complete your profile first for a personalized chat.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Check for and display image
            if "image" in message and message["image"] is not None:
                st.image(message["image"], width=200)
            # Check for and display text
            if "text" in message and message["text"]:
                st.write(message["text"])

    # --- New Input area ---
    user_lang_code = st.session_state.profile.get("language", "en")
    user_lang_name = SUPPORTED_LANGUAGES.get(user_lang_code, "your language")
    
    # We use st.container to group the uploader and the chat_input
    with st.container():
        uploaded_image = st.file_uploader("Upload an image of your crop (optional)", type=["png", "jpg", "jpeg"])
        query = st.chat_input(f"Ask a question in {user_lang_name}...")

    if query:
        # Prepare user message for history
        user_message = {"role": "user", "text": query}
        image_bytes = None
        if uploaded_image is not None:
            image_bytes = uploaded_image.getvalue()
            user_message["image"] = image_bytes

        # Append user message and display it
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            if image_bytes:
                st.image(image_bytes, width=200)
            st.write(query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Krishi Sakhi is analyzing and thinking..."):
                response_text = process_chat_query(
                    query=query,
                    profile=st.session_state.profile,
                    predictions=st.session_state.predictions,
                    faiss_index=faiss_index,
                    docs=docs,
                    language=user_lang_code,
                    image_bytes=image_bytes  # Pass the image data to the backend
                )
                st.write(response_text)
                # Append assistant message to history
                st.session_state.chat_history.append({"role": "assistant", "text": response_text})

if __name__ == "__main__":
    main()
