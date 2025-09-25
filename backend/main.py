from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import time
import json
import traceback
import sys
import uuid

# Add the app directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from enhanced_pdf_to_md import pdf_to_markdown
from chunks import create_retriever
from qa_graph import build_graph as build_qa_graph

app = FastAPI(title="AI Assistant API", version="1.0.0")

# CORS middleware to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
faiss_db_path = "faiss_db"
files_path = "files"

# Pydantic models for request/response
class ChatMessage(BaseModel):
    content: str
    conversation_history: Optional[List[tuple]] = []

class ChatResponse(BaseModel):
    content: str
    response_time: float

class StatusResponse(BaseModel):
    status: str
    message: str

class DatabaseStatus(BaseModel):
    exists: bool
    file_count: int


# Helper functions
def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(files_path, exist_ok=True)
    return True

def get_database_status():
    """Check if FAISS database exists and get file count"""
    exists = os.path.exists(faiss_db_path)
    file_count = 0
    if os.path.exists(files_path):
        file_count = len([f for f in os.listdir(files_path) if f.endswith('.pdf')])
    return DatabaseStatus(exists=exists, file_count=file_count)


# API Routes

@app.get("/")
async def root():
    return {"message": "ai assistant api"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/database/status")
async def database_status():
    """Get the current database status"""
    return get_database_status()


@app.post("/database/reset")
async def reset_database():
    """Reset the database and clear all files"""
    try:
        if os.path.exists(faiss_db_path):
            shutil.rmtree(faiss_db_path)
        if os.path.exists(files_path):
            shutil.rmtree(files_path)
        ensure_directories()
        return StatusResponse(status="success", message="database reset successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to reset database: {str(e)}")

@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload PDF documents and create database"""
    try:
        print(f"\n=== DOCUMENT UPLOAD START ===")
        print(f"Number of files received: {len(files)}")

        upload_start = time.time()
        ensure_directories()
        print(f"Directories ensured: {files_path}, {faiss_db_path}")

        # Save uploaded files
        uploaded_files = []
        for i, file in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {file.filename}")

            if not file.filename.endswith('.pdf'):
                print(f"ERROR: Invalid file type for {file.filename}")
                raise HTTPException(status_code=400, detail=f"only pdf files allowed: {file.filename}")

            file_path = os.path.join(files_path, file.filename)
            file_size = 0
            with open(file_path, "wb") as f:
                content = await file.read()
                file_size = len(content)
                f.write(content)
            uploaded_files.append(file_path)
            print(f"  - Saved: {file.filename} ({file_size:,} bytes)")

        upload_time = time.time() - upload_start
        print(f"File upload completed in {upload_time:.3f}s")

        # Process PDFs to markdown
        print("Converting PDFs to markdown...")
        conversion_start = time.time()
        for i, pdf_path in enumerate(uploaded_files):
            print(f"Converting PDF {i+1}/{len(uploaded_files)}: {os.path.basename(pdf_path)}")
            pdf_to_markdown(pdf_path)
            print(f"  - Converted: {os.path.basename(pdf_path)}")

        conversion_time = time.time() - conversion_start
        print(f"PDF conversion completed in {conversion_time:.3f}s")

        # Create retriever (this builds the FAISS database)
        print("Building FAISS database...")
        db_start = time.time()
        create_retriever(files_path, faiss_db_path)
        db_time = time.time() - db_start
        print(f"FAISS database created in {db_time:.3f}s")

        total_time = time.time() - upload_start
        print(f"Total upload process time: {total_time:.3f}s")
        print(f"=== DOCUMENT UPLOAD END ===\n")

        return StatusResponse(status="success", message=f"uploaded {len(uploaded_files)} files and created database")

    except Exception as e:
        print(f"ERROR in upload_documents: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to upload documents: {str(e)}")

@app.post("/qa/chat")
async def qa_chat(message: ChatMessage):
    """Process Q&A chat message"""
    try:
        print(f"\n=== QA CHAT REQUEST START ===")
        print(f"Question: {message.content}")
        print(f"Conversation History: {len(message.conversation_history)} messages")

        if not os.path.exists(faiss_db_path):
            print(f"ERROR: FAISS database not found at {faiss_db_path}")
            raise HTTPException(status_code=400, detail="no database found. upload documents first.")

        start_time = time.time()
        print(f"Request start time: {start_time}")

        # Build QA graph with retriever
        print("Building QA graph and retriever...")
        retriever_start = time.time()
        retriever = create_retriever(files_path, faiss_db_path)
        app_graph = build_qa_graph(retriever)
        retriever_time = time.time() - retriever_start
        print(f"QA graph built in {retriever_time:.3f}s")

        inputs = {
            "question": message.content,
            "conversation_history": message.conversation_history
        }
        print(f"Graph inputs prepared: {inputs}")

        # Stream through the graph
        print("Starting graph execution...")
        graph_start = time.time()
        full_response = ""

        final_state = {}
        step_count = 0
        for output in app_graph.stream(inputs):
            step_count += 1
            print(f"Graph step {step_count}: {list(output.keys())}")
            for key, value in output.items():
                if key == "generate":
                    final_state = value
                    print(f"Final generation received (length: {len(value.get('generation', ''))} chars)")

        graph_time = time.time() - graph_start
        print(f"Graph execution completed in {graph_time:.3f}s ({step_count} steps)")

        if final_state:
            full_response = final_state.get("generation", "")
            print(f"Response extracted (length: {len(full_response)} chars)")

        if not full_response:
            full_response = "i couldn't generate a response. please try rephrasing your question."
            print("WARNING: No response generated, using fallback message")

        end_time = time.time()
        response_time = end_time - start_time
        print(f"Total request time: {response_time:.3f}s")
        print(f"=== QA CHAT REQUEST END ===\n")

        return ChatResponse(
            content=full_response,
            response_time=response_time
        )

    except Exception as e:
        print(f"ERROR in qa_chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"chat processing failed: {str(e)}")

@app.post("/qa/chat/stream")
async def qa_chat_stream(message: ChatMessage):
    """Process Q&A chat message with streaming response"""
    try:
        print(f"\n=== STREAMING CHAT REQUEST START ===")
        print(f"Question: {message.content}")
        print(f"Conversation History: {len(message.conversation_history)} messages")

        if not os.path.exists(faiss_db_path):
            print(f"ERROR: FAISS database not found at {faiss_db_path}")
            raise HTTPException(status_code=400, detail="no database found. upload documents first.")

        # Build QA graph with retriever
        print("Building QA graph and retriever for streaming...")
        setup_start = time.time()
        retriever = create_retriever(files_path, faiss_db_path)
        app_graph = build_qa_graph(retriever)
        setup_time = time.time() - setup_start
        print(f"QA graph built in {setup_time:.3f}s")

        inputs = {
            "question": message.content,
            "conversation_history": message.conversation_history
        }
        print(f"Stream inputs prepared: {inputs}")

        async def generate_stream():
            try:
                start_time = time.time()
                print("Starting streaming response generation...")

                # First get documents (retrieval phase)
                print("Retrieving relevant documents...")
                retrieval_start = time.time()
                docs = retriever.get_relevant_documents(message.content)
                retrieval_time = time.time() - retrieval_start
                print(f"Retrieved {len(docs)} documents in {retrieval_time:.3f}s")

                documents_text = ""
                for i, doc in enumerate(docs):
                    documents_text += doc.page_content
                    print(f"  Doc {i+1}: {len(doc.page_content)} chars from {doc.metadata.get('file_name', 'Unknown')}")

                print(f"Total document context: {len(documents_text)} characters")

                # Stream the LLM response directly
                print("Starting LLM streaming...")
                stream_start = time.time()
                from llm_nodes import generator
                stream_generator = generator.stream({
                    "conversation_history": message.conversation_history,
                    "document": documents_text,
                    "question": message.content
                })

                response_content = ""
                chunk_count = 0
                for chunk in stream_generator:
                    if chunk:
                        chunk_count += 1
                        response_content += chunk
                        print(f"Streaming chunk {chunk_count}: '{chunk}' ({len(chunk)} chars)")
                        # Send each chunk as it arrives
                        yield f"data: {json.dumps({'content': chunk, 'type': 'chunk'})}\n\n"

                stream_time = time.time() - stream_start
                print(f"LLM streaming completed in {stream_time:.3f}s ({chunk_count} chunks)")

                # Add sources after the main content
                print("Adding source information...")
                from qa_graph import format_sources_for_display
                source_info = format_sources_for_display(docs)
                if source_info:
                    print(f"Source info: {source_info}")
                    yield f"data: {json.dumps({'content': source_info, 'type': 'chunk'})}\n\n"

                # Send completion signal with total time
                end_time = time.time()
                response_time = end_time - start_time
                print(f"Total streaming response time: {response_time:.3f}s")
                print(f"Final response length: {len(response_content)} characters")
                print("=== STREAMING CHAT REQUEST END ===\n")
                yield f"data: {json.dumps({'type': 'complete', 'response_time': response_time})}\n\n"

            except Exception as e:
                print(f"ERROR in streaming generator: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"streaming chat failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    ensure_directories()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)