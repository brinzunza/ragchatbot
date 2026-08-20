from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import shutil
import time
import json
import traceback
import sys
import uuid

# Add the app directory to the path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from pdf_converter import pdf_to_markdown
from document_processor import get_retriever, index_documents, list_documents, delete_document, has_documents, clear_vectorstore_cache
from qa_workflow import build_graph as build_qa_graph
import chat_manager
import llm_config

app = FastAPI(title="AI Assistant API", version="1.0.0")

# CORS middleware to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload progress tracking - upload_id is globally unique (uuid4), so this
# doesn't need to be scoped by chat_id to work correctly. Each entry also
# records which chat it belongs to, so create_chat() can refuse to clone a
# chat that's mid-upload.
upload_progress: Dict[str, Dict] = {}

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

class UploadResponse(BaseModel):
    upload_id: str
    status: str
    message: str

class UploadProgress(BaseModel):
    upload_id: str
    status: str  # 'uploading', 'converting', 'indexing', 'completed', 'error'
    progress: int  # 0-100
    message: str
    current_file: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0

class DocumentInfo(BaseModel):
    name: str
    chunk_count: int
    uploaded_at: str

class ChatInfo(BaseModel):
    id: str
    name: str
    created_at: str
    document_count: int

class CreateChatRequest(BaseModel):
    name: str
    copy_from: Optional[str] = None

class ApiKeyStatus(BaseModel):
    has_key: bool
    masked_key: Optional[str] = None
    provider: str
    model: str

class SetApiKeyRequest(BaseModel):
    api_key: str

class SaveMessagesRequest(BaseModel):
    messages: List[Dict]


# Helper functions
def get_database_status(chroma_db_path: str, files_path: str):
    """Check if any documents are indexed and get file count"""
    exists = has_documents(chroma_db_path)
    file_count = len(list_documents(chroma_db_path, files_path)) if exists else 0
    return DatabaseStatus(exists=exists, file_count=file_count)

def in_progress_upload_chat_ids():
    """Chat ids with an upload currently converting/indexing/uploading"""
    return {
        v["chat_id"] for v in upload_progress.values()
        if v.get("chat_id") and v["status"] not in ("completed", "error")
    }

def resolve_chat_paths(chat_id: str):
    """get_chat_paths, translating a missing chat into a 404"""
    try:
        return chat_manager.get_chat_paths(chat_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}")

def process_documents_background(upload_id: str, file_paths: List[str], chroma_db_path: str, files_path: str):
    """Background task to process uploaded documents"""
    try:
        total_files = len(file_paths)
        print(f"\n=== BACKGROUND PROCESSING START (ID: {upload_id}) ===")

        # Stage 1: Convert PDFs to markdown
        upload_progress[upload_id].update({
            "status": "converting",
            "progress": 10,
            "message": "converting PDFs to markdown",
            "current_file": None,
            "processed_files": 0
        })

        print("Converting PDFs to markdown...")
        conversion_start = time.time()
        for i, pdf_path in enumerate(file_paths):
            filename = os.path.basename(pdf_path)
            print(f"Converting PDF {i+1}/{total_files}: {filename}")

            upload_progress[upload_id]["current_file"] = filename
            upload_progress[upload_id]["processed_files"] = i
            upload_progress[upload_id]["progress"] = 10 + int((i / total_files) * 40)

            pdf_to_markdown(pdf_path, output_dir=files_path)
            print(f"  - Converted: {filename}")

        conversion_time = time.time() - conversion_start
        print(f"PDF conversion completed in {conversion_time:.3f}s")

        # Stage 2: Build vector database
        upload_progress[upload_id].update({
            "status": "indexing",
            "progress": 50,
            "message": "building vector database",
            "current_file": None,
            "processed_files": total_files
        })

        def indexing_progress(current_chunk, total_chunks, file_name):
            pct = int((current_chunk / total_chunks) * 40) if total_chunks else 0
            upload_progress[upload_id].update({
                "status": "indexing",
                "progress": 50 + pct,
                "message": f"summarizing chunk {current_chunk}/{total_chunks}",
                "current_file": file_name,
                "processed_files": total_files
            })

        print("Building vector database...")
        db_start = time.time()
        index_documents(files_path, chroma_db_path, progress_callback=indexing_progress)
        db_time = time.time() - db_start
        print(f"Vector database updated in {db_time:.3f}s")

        # Stage 3: Cleanup - delete PDF files to save storage
        upload_progress[upload_id].update({
            "status": "cleanup",
            "progress": 90,
            "message": "cleaning up temporary files",
            "current_file": None,
            "processed_files": total_files
        })

        print("Cleaning up PDF files...")
        cleanup_start = time.time()
        deleted_count = 0
        for pdf_path in file_paths:
            try:
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path)
                    os.remove(pdf_path)
                    deleted_count += 1
                    print(f"  - Deleted: {os.path.basename(pdf_path)} ({file_size:,} bytes)")
            except Exception as e:
                print(f"  - Warning: Could not delete {os.path.basename(pdf_path)}: {str(e)}")

        cleanup_time = time.time() - cleanup_start
        print(f"Cleanup completed in {cleanup_time:.3f}s ({deleted_count}/{total_files} files deleted)")

        # Complete
        upload_progress[upload_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"successfully processed {total_files} files",
            "current_file": None,
            "processed_files": total_files
        })

        print(f"=== BACKGROUND PROCESSING COMPLETE (ID: {upload_id}) ===\n")

    except Exception as e:
        print(f"ERROR in background processing: {str(e)}")
        traceback.print_exc()
        upload_progress[upload_id].update({
            "status": "error",
            "progress": 0,
            "message": f"processing failed: {str(e)}",
            "current_file": None,
            "processed_files": 0
        })


# API Routes

@app.get("/")
async def root():
    return {"message": "ai assistant api"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# --- Settings (global, shared across all chats) ---

@app.get("/settings/api-key")
async def get_api_key_status():
    """Report whether an OpenAI key is set and which provider is active"""
    info = llm_config.get_model_info()
    masked = llm_config.get_masked_api_key()
    return ApiKeyStatus(has_key=masked is not None, masked_key=masked, **info)

@app.post("/settings/api-key")
async def set_api_key(request: SetApiKeyRequest):
    """Set (or replace) the OpenAI API key, applied immediately without a restart"""
    key = request.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="api_key cannot be empty")
    try:
        llm_config.set_api_key(key)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to save api key: {str(e)}")
    info = llm_config.get_model_info()
    return ApiKeyStatus(has_key=True, masked_key=llm_config.get_masked_api_key(), **info)

@app.delete("/settings/api-key")
async def remove_api_key():
    """Remove the OpenAI key, falling back to local Ollama immediately"""
    try:
        llm_config.clear_api_key()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to remove api key: {str(e)}")
    info = llm_config.get_model_info()
    return ApiKeyStatus(has_key=False, masked_key=None, **info)


# --- Chat management ---

@app.get("/chats")
async def get_chats():
    """List all chats with their document counts"""
    try:
        return [ChatInfo(**c) for c in chat_manager.list_chats()]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to list chats: {str(e)}")

@app.post("/chats")
async def create_chat(request: CreateChatRequest):
    """Create a new chat, empty or cloned from an existing one's documents"""
    try:
        chat = chat_manager.create_chat(
            request.name,
            copy_from=request.copy_from,
            in_progress_upload_chat_ids=in_progress_upload_chat_ids()
        )
        return ChatInfo(**chat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to create chat: {str(e)}")

@app.delete("/chats/{chat_id}")
async def remove_chat(chat_id: str):
    """Delete a chat and all its documents"""
    try:
        deleted = chat_manager.delete_chat(chat_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}")
        return StatusResponse(status="success", message=f"deleted chat {chat_id}")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to delete chat: {str(e)}")

@app.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    """Load a chat's persisted conversation history"""
    try:
        return {"messages": chat_manager.get_messages(chat_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to load messages: {str(e)}")

@app.put("/chats/{chat_id}/messages")
async def save_messages(chat_id: str, request: SaveMessagesRequest):
    """Persist a chat's full conversation history"""
    try:
        chat_manager.save_messages(chat_id, request.messages)
        return StatusResponse(status="success", message="messages saved")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to save messages: {str(e)}")


# --- Per-chat document database ---

@app.get("/chats/{chat_id}/database/status")
async def database_status(chat_id: str):
    """Get the current database status for a chat"""
    chroma_db_path, files_path = resolve_chat_paths(chat_id)
    return get_database_status(chroma_db_path, files_path)

@app.post("/chats/{chat_id}/database/reset")
async def reset_database(chat_id: str):
    """Wipe a chat's documents, keeping the chat itself"""
    chroma_db_path, files_path = resolve_chat_paths(chat_id)
    try:
        if os.path.exists(chroma_db_path):
            shutil.rmtree(chroma_db_path)
            clear_vectorstore_cache()
        if os.path.exists(files_path):
            shutil.rmtree(files_path)
        os.makedirs(files_path, exist_ok=True)
        return StatusResponse(status="success", message="database reset successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to reset database: {str(e)}")

@app.get("/chats/{chat_id}/documents")
async def get_documents(chat_id: str):
    """List indexed documents with chunk counts and upload dates"""
    chroma_db_path, files_path = resolve_chat_paths(chat_id)
    try:
        docs = list_documents(chroma_db_path, files_path)
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to list documents: {str(e)}")

@app.get("/chats/{chat_id}/documents/{name}/content")
async def get_document_content(chat_id: str, name: str):
    """Get the raw markdown content of a document, for citation highlighting"""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid document name")
    _, files_path = resolve_chat_paths(chat_id)
    try:
        path = os.path.join(files_path, f"{name}.md")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"document not found: {name}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"name": name, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to read document: {str(e)}")

@app.delete("/chats/{chat_id}/documents/{name}")
async def remove_document(chat_id: str, name: str):
    """Delete a single document from the index"""
    chroma_db_path, files_path = resolve_chat_paths(chat_id)
    try:
        deleted = delete_document(name, chroma_db_path, files_path)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"document not found: {name}")
        os.makedirs(files_path, exist_ok=True)
        return StatusResponse(status="success", message=f"deleted {name}")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to delete document: {str(e)}")

@app.post("/chats/{chat_id}/documents/upload")
async def upload_documents(chat_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDF documents to a chat and process in background"""
    chroma_db_path, files_path = resolve_chat_paths(chat_id)
    try:
        print(f"\n=== DOCUMENT UPLOAD START ===")
        print(f"Number of files received: {len(files)}")

        upload_start = time.time()
        os.makedirs(files_path, exist_ok=True)

        # Generate upload ID
        upload_id = str(uuid.uuid4())
        print(f"Upload ID: {upload_id}")

        # Initialize progress tracking
        upload_progress[upload_id] = {
            "chat_id": chat_id,
            "status": "uploading",
            "progress": 0,
            "message": "receiving files",
            "current_file": None,
            "total_files": len(files),
            "processed_files": 0
        }

        # Save uploaded files
        uploaded_files = []
        for i, file in enumerate(files):
            print(f"Receiving file {i+1}/{len(files)}: {file.filename}")

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

            # Update progress
            upload_progress[upload_id]["progress"] = int((i + 1) / len(files) * 10)
            upload_progress[upload_id]["current_file"] = file.filename
            print(f"  - Saved: {file.filename} ({file_size:,} bytes)")

        upload_time = time.time() - upload_start
        print(f"File upload completed in {upload_time:.3f}s")
        print(f"Starting background processing for {len(uploaded_files)} files")

        # Start background processing
        background_tasks.add_task(process_documents_background, upload_id, uploaded_files, chroma_db_path, files_path)

        return UploadResponse(
            upload_id=upload_id,
            status="processing",
            message=f"received {len(uploaded_files)} files, processing in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in upload_documents: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"failed to upload documents: {str(e)}")

@app.get("/documents/upload/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get the status of an upload/processing job"""
    if upload_id not in upload_progress:
        raise HTTPException(status_code=404, detail="upload not found")

    progress_data = upload_progress[upload_id]
    return UploadProgress(
        upload_id=upload_id,
        status=progress_data["status"],
        progress=progress_data["progress"],
        message=progress_data["message"],
        current_file=progress_data.get("current_file"),
        total_files=progress_data["total_files"],
        processed_files=progress_data["processed_files"]
    )

@app.post("/chats/{chat_id}/qa/chat")
async def qa_chat(chat_id: str, message: ChatMessage):
    """Process Q&A chat message"""
    chroma_db_path, _ = resolve_chat_paths(chat_id)
    try:
        print(f"\n=== QA CHAT REQUEST START ===")
        print(f"Question: {message.content}")
        print(f"Conversation History: {len(message.conversation_history)} messages")

        if not has_documents(chroma_db_path):
            print(f"ERROR: no documents indexed at {chroma_db_path}")
            raise HTTPException(status_code=400, detail="no database found. upload documents first.")

        start_time = time.time()
        print(f"Request start time: {start_time}")

        # Build QA graph with retriever
        print("Building QA graph and retriever...")
        retriever_start = time.time()
        retriever = get_retriever(chroma_db_path, top_k=3)
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in qa_chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"chat processing failed: {str(e)}")

@app.post("/chats/{chat_id}/qa/chat/stream")
async def qa_chat_stream(chat_id: str, message: ChatMessage):
    """Process Q&A chat message with streaming response"""
    chroma_db_path, _ = resolve_chat_paths(chat_id)
    try:
        print(f"\n=== STREAMING CHAT REQUEST START ===")
        print(f"Question: {message.content}")
        print(f"Conversation History: {len(message.conversation_history)} messages")

        if not has_documents(chroma_db_path):
            print(f"ERROR: no documents indexed at {chroma_db_path}")
            raise HTTPException(status_code=400, detail="no database found. upload documents first.")

        # Build QA graph with retriever
        print("Building QA graph and retriever for streaming...")
        setup_start = time.time()
        retriever = get_retriever(chroma_db_path, top_k=3)
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
                docs = retriever.invoke(message.content)
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
                from llm_prompts import generator
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

                # Add structured sources after the main content - one entry per
                # retrieved chunk (not deduplicated by file) so the frontend can
                # cite and highlight every chunk that contributed to the answer
                print("Adding source information...")
                from qa_workflow import get_sources_metadata
                sources = get_sources_metadata(docs)
                if sources:
                    print(f"Source info: {sources}")
                    yield f"data: {json.dumps({'content': sources, 'type': 'sources'})}\n\n"

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

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"streaming chat failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    os.makedirs(chat_manager.CHATS_ROOT, exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
