import streamlit as st
import os
import shutil
import time
from enhanced_pdf_to_md import pdf_to_markdown
from chunks import create_retriever
from qa_graph import build_graph as build_qa_graph
from da_graph import build_da_graph
import pandas as pd
import re
import json
from PIL import Image
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure Streamlit page settings
st.set_page_config(
    page_title="AI Document Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-minimalistic CSS with JetBrains Mono
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap');

    /* Global reset and font */
    * {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 300 !important;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main container - centered and minimal */
    .stApp {
        background-color: #ffffff;
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    .main .block-container {
        padding: 0;
        max-width: 100%;
    }

    /* Sidebar - minimal and centered */
    .css-1d391kg {
        background-color: #ffffff;
        border: none;
        text-align: center;
        padding: 2rem 1rem;
        width: 200px !important;
        margin: 0 auto;
    }

    /* All text lowercase and minimal */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        text-transform: lowercase !important;
        color: #000000 !important;
        font-size: 12px !important;
        font-weight: 300 !important;
        line-height: 1.4 !important;
        margin: 0.5rem 0 !important;
    }

    /* Main titles */
    .main-title {
        font-size: 16px !important;
        font-weight: 400 !important;
        margin: 2rem 0 1rem 0 !important;
        text-align: center;
    }

    /* Text-like buttons */
    .stButton > button {
        background: none !important;
        border: none !important;
        color: #000000 !important;
        font-size: 12px !important;
        font-weight: 300 !important;
        text-transform: lowercase !important;
        padding: 0 !important;
        margin: 0.25rem 0 !important;
        cursor: pointer;
        text-decoration: none;
        transition: text-decoration 0.2s ease;
    }

    .stButton > button:hover {
        text-decoration: underline !important;
        background: none !important;
        border: none !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Radio buttons - minimal */
    .stRadio {
        text-align: center;
    }

    .stRadio > div {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 1rem 0 !important;
    }

    .stRadio label {
        font-size: 12px !important;
        color: #000000 !important;
        cursor: pointer;
        margin: 0.5rem 0 !important;
    }

    .stRadio label:hover {
        text-decoration: underline;
    }

    /* Chat messages */
    .stChatMessage {
        background: none !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        margin: 1rem 0 !important;
    }

    /* Chat input */
    .stChatInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        padding: 0.5rem !important;
        font-size: 12px !important;
        color: #000000 !important;
        text-align: center;
    }

    .stChatInput > div > div > input:focus {
        border: 1px solid #000000 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* File uploader */
    .stFileUploader {
        border: 1px solid #000000 !important;
        background: #ffffff !important;
        border-radius: 0 !important;
        padding: 1rem !important;
        text-align: center;
    }

    .stFileUploader:hover {
        background: #ffffff !important;
        border: 1px solid #000000 !important;
    }

    /* Status messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        color: #000000 !important;
        text-align: center;
        padding: 0.5rem !important;
    }

    /* Minimal cards */
    .minimal-card {
        text-align: center;
        padding: 1rem 0;
        margin: 1rem 0;
    }

    /* Remove all decorative elements */
    .stSpinner, .stProgress, hr {
        display: none;
    }

    /* Caption styling */
    .caption {
        color: #000000 !important;
        font-size: 10px !important;
        text-align: center;
        margin: 0.5rem 0 !important;
    }

    /* Center everything */
    .stContainer, .stColumn {
        text-align: center;
    }

    /* Minimal spacing */
    .block-container {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

faiss_db_path = "faiss_db"

# Ultra-minimal sidebar
with st.sidebar:
    st.markdown('<div class="minimal-card"><p>ai assistant</p></div>', unsafe_allow_html=True)

    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Document Q&A"

    # Minimal mode selection
    mode_options = {
        "document q&a": "Document Q&A",
        "data analysis": "Advanced Data Analysis"
    }

    selected_mode = st.radio(
        "mode",
        options=list(mode_options.keys()),
        index=list(mode_options.values()).index(st.session_state.app_mode),
        label_visibility="collapsed"
    )

    st.session_state.app_mode = mode_options[selected_mode]

    st.markdown('<div class="minimal-card"></div>', unsafe_allow_html=True)

    # Minimal controls
    if st.button("clear chat"):
        if st.session_state.app_mode == "Document Q&A":
            if "qa_messages" in st.session_state:
                st.session_state.qa_messages = []
        elif st.session_state.app_mode == "Advanced Data Analysis":
            if "analysis_messages" in st.session_state:
                st.session_state.analysis_messages = []
        st.rerun()

    if st.session_state.app_mode == "Document Q&A":
        if st.button("reset database"):
            if os.path.exists(faiss_db_path):
                shutil.rmtree(faiss_db_path)
            if os.path.exists("files"):
                shutil.rmtree("files")
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    st.markdown('<div class="minimal-card"></div>', unsafe_allow_html=True)

    # Minimal mode info
    if st.session_state.app_mode == "Document Q&A":
        st.markdown('<div class="minimal-card"><p>pdf document analysis</p></div>', unsafe_allow_html=True)
    elif st.session_state.app_mode == "Advanced Data Analysis":
        st.markdown('<div class="minimal-card"><p>csv data analysis</p></div>', unsafe_allow_html=True)

# Document Q&A Mode Interface
if st.session_state.app_mode == "Document Q&A":
    st.markdown('<div class="main-title">document q&a</div>', unsafe_allow_html=True)

    # Database initialization flow
    if not os.path.exists(faiss_db_path):
        st.markdown('<div class="minimal-card"><p>upload pdf documents to begin</p></div>', unsafe_allow_html=True)

        if not os.path.exists("files"):
            os.makedirs("files")

        uploaded_files = st.file_uploader(
            "select pdf files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="visible"
        )

        if uploaded_files:
            with st.spinner("Processing PDFs and building the knowledge base... This may take a moment."):
                for uploaded_file in uploaded_files:
                    pdf_file_path = os.path.join("files", uploaded_file.name)
                    with open(pdf_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_to_markdown(pdf_file_path)

                create_retriever("files", faiss_db_path)

            st.markdown('<div class="minimal-card"><p>database created</p></div>', unsafe_allow_html=True)
            st.rerun()
        
        prompt = st.chat_input("upload pdf files first")
        if prompt:
            st.markdown('<div class="minimal-card"><p>upload files before asking questions</p></div>', unsafe_allow_html=True)

    else:
        # Main QA chat interface
        if "qa_messages" not in st.session_state:
            st.session_state.qa_messages = []

        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.qa_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "images" in message and message["images"]:
                        for image_path in message["images"]:
                            if os.path.exists(image_path) and os.path.isfile(image_path):
                                try:
                                    with Image.open(image_path) as img:
                                        st.image(image_path, caption=os.path.basename(image_path))
                                except:
                                    pass
                    if "response_time" in message:
                        st.markdown(f'<div class="caption">⏱️ Response time: {message["response_time"]:.2f}s</div>', unsafe_allow_html=True)

        # Process user input and generate response
        prompt = st.chat_input("ask about your documents")
        if prompt:
            st.session_state.qa_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                start_time = time.time()
                
                with st.spinner("Thinking..."):
                    context_window_size = 3
                    messages_for_context = st.session_state.qa_messages[:-1]
                    
                    conversation_history = []
                    for i in range(len(messages_for_context) - 2, -1, -2):
                        if len(conversation_history) < context_window_size:
                            if messages_for_context[i]['role'] == 'user' and messages_for_context[i+1]['role'] == 'assistant':
                                question = messages_for_context[i]['content']
                                answer = messages_for_context[i+1]['content']
                                conversation_history.insert(0, (question, answer))
                        else:
                            break

                    retriever = create_retriever("files", faiss_db_path)
                    app = build_qa_graph(retriever)

                    inputs = {
                        "question": prompt,
                        "conversation_history": conversation_history
                    }
                    
                    full_response = ""
                    response_image_paths = []
                    
                    print(f"PROCESSING QUESTION: {prompt}")

                    final_state = {}
                    for output in app.stream(inputs):
                        for key, value in output.items():
                            if key == "generate":
                                final_state = value

                    if final_state:
                        full_response = final_state.get("generation", "")
                        response_image_paths = final_state.get("image_paths", [])
                
                # Display response with typing effect and images
                if full_response or response_image_paths:
                    if full_response:
                        message_placeholder = st.empty()
                        displayed_text = ""
                        for char in full_response:
                            displayed_text += char
                            message_placeholder.markdown(displayed_text + "▌")
                            time.sleep(0.01)
                        message_placeholder.markdown(full_response)
                    
                    if response_image_paths:
                        for image_path in response_image_paths:
                            try:
                                if os.path.exists(image_path) and os.path.isfile(image_path):
                                    with Image.open(image_path) as img:
                                        st.image(image_path, caption=f"Source: {os.path.basename(image_path)}")
                            except Exception:
                                pass

                    end_time = time.time()
                    response_time = end_time - start_time
                    st.markdown(f'<div class="caption">{response_time:.2f}s</div>', unsafe_allow_html=True)

                    st.session_state.qa_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "images": response_image_paths,
                        "response_time": response_time
                    })
                else:
                    error_message = "I apologize, but I couldn't generate a response to your question. Please try rephrasing your question or ask something else."
                    st.markdown(error_message)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.markdown(f'<div class="caption">{response_time:.2f}s</div>', unsafe_allow_html=True)

                    st.session_state.qa_messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "response_time": response_time
                    })
            
            st.rerun()

# Advanced Data Analysis Mode Interface
elif st.session_state.app_mode == "Advanced Data Analysis":
    st.markdown('<div class="main-title">data analysis</div>', unsafe_allow_html=True)

    if "analysis_messages" not in st.session_state:
        st.session_state.analysis_messages = []

    data_file_path = "files/clean_data.csv"

    if os.path.exists(data_file_path):
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.analysis_messages:
                with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                    
                if "response_time" in message:
                    st.markdown(f'<div class="caption">⏱️ Response time: {message["response_time"]:.2f}s</div>', unsafe_allow_html=True)

        # Process user input for data analysis
        prompt = st.chat_input("ask about your data")
        if prompt:
            st.session_state.analysis_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    start_time = time.time()
                    
                    try:
                        app = build_da_graph(data_file_path)
                        
                        inputs = {"question": prompt}
                        
                        final_state = {}
                        for output in app.stream(inputs):
                            final_state = output
                        full_response = final_state.get('analyze_results').get('generation')
                        
                        if full_response:
                            st.markdown(full_response)
                        else:
                            st.error("No response generated from the analysis.")

                        end_time = time.time()
                        response_time = end_time - start_time
                        st.markdown(f'<div class="caption">{response_time:.2f}s</div>', unsafe_allow_html=True)

                        message_obj = {
                            "role": "assistant",
                            "content": full_response,
                            "response_time": response_time
                        }
                        st.session_state.analysis_messages.append(message_obj)

                    except Exception as e:
                        end_time = time.time()
                        response_time = end_time - start_time
                        error_message = f"Error during data analysis: {str(e)}"
                        st.error(error_message)
                        
                        st.session_state.analysis_messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "response_time": response_time
                        })
                        
                        print(f"ERROR in analysis: {e}")
                        import traceback
                        traceback.print_exc()
            
            st.rerun()
    else:
        st.markdown('''
        <div class="minimal-card">
            <p>no data file found</p>
            <p>csv files with structured data expected</p>
        </div>
        ''', unsafe_allow_html=True)

        prompt = st.chat_input("upload csv file first")
        if prompt:
            st.markdown('<div class="minimal-card"><p>upload csv file before asking questions</p></div>', unsafe_allow_html=True)

