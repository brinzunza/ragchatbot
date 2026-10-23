#!/bin/bash

# Start the FastAPI backend
echo "starting ai assistant backend..."

# Check if we're in a virtual environment, if not create one
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "creating virtual environment for backend..."
    python3 -m venv backend_env
    source backend_env/bin/activate
fi

# Install dependencies
echo "installing backend dependencies..."
cd backend
pip install -r requirements.txt

# Install the original app dependencies (from the main app)
cd ../
pip install streamlit pandas chromadb "langchain-chroma==0.2.6" langchain langchain-community langchain-nomic

# Start the FastAPI server
echo "starting fastapi server on http://localhost:8000 (this machine only)"
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
