from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
import os
import glob
import time
from typing import List
from langchain_community.chat_models import ChatOllama

# Main retriever creation function that processes markdown files into searchable chunks
def create_retriever(
    md_folder_path: str,
    faiss_db_path: str = "faiss_db",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    top_k: int = 3
):
    print(f"\n--- CREATE_RETRIEVER START ---")
    print(f"MD folder path: {md_folder_path}")
    print(f"FAISS DB path: {faiss_db_path}")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}, top_k: {top_k}")

    embedding_start = time.time()
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    embedding_init_time = time.time() - embedding_start
    print(f"Embedding model initialized in {embedding_init_time:.3f}s")

    # Load existing FAISS database if it exists, otherwise create a new one
    if os.path.exists(faiss_db_path):
        print(f"Loading existing FAISS database from {faiss_db_path}")
        load_start = time.time()
        vectorstore = FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
        load_time = time.time() - load_start
        print(f"FAISS database loaded in {load_time:.3f}s")
    else:
        print(f"Creating new FAISS database...")
        create_start = time.time()

        md_files = glob.glob(os.path.join(md_folder_path, "*.md"))
        print(f"Found {len(md_files)} markdown files: {[os.path.basename(f) for f in md_files]}")

        if not md_files:
            raise ValueError(f"No .md files found in {md_folder_path}")

        all_documents = []

        # Load and process all markdown files
        file_load_start = time.time()
        for i, md_file in enumerate(md_files):
            print(f"Processing file {i+1}/{len(md_files)}: {os.path.basename(md_file)}")
            try:
                loader = TextLoader(md_file, encoding='utf-8')
                documents = loader.load()

                for doc in documents:
                    file_name = os.path.basename(md_file)
                    doc_size = len(doc.page_content)

                    doc.metadata.update({
                        'file_name': file_name,
                        'source_file': md_file
                    })
                    print(f"  - Document loaded: {doc_size} characters")

                all_documents.extend(documents)
                print(f"  - Completed: {file_name} ({len(documents)} document(s))")

            except Exception as e:
                print(f"  ERROR loading {md_file}: {e}")
                continue

        file_load_time = time.time() - file_load_start
        print(f"All files loaded in {file_load_time:.3f}s (total docs: {len(all_documents)})")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded")

        # Split documents into searchable chunks
        print("Splitting documents into chunks...")
        split_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        texts = text_splitter.split_documents(all_documents)
        split_time = time.time() - split_start
        print(f"Document splitting completed in {split_time:.3f}s ({len(texts)} chunks created)")

        # Enhance chunks with AI-generated summaries
        print("Enhancing chunks with AI-generated summaries...")
        summary_start = time.time()
        for i, chunk in enumerate(texts):
            surrounding_chunks = texts[max(0, i-1):i] + texts[i+1:i+2]
            surrounding_text = "\n\n".join([c.page_content for c in surrounding_chunks])

            amount = len(texts)
            chunk_start = time.time()
            print(f"Processing chunk {i+1}/{amount} from {os.path.basename(chunk.metadata['source_file'])}")

            summary = generate_summary_with_ollama(chunk.page_content, surrounding_text, chunk.metadata['source_file'])
            chunk_time = time.time() - chunk_start

            original_size = len(chunk.page_content)
            chunk.metadata = {
                'file_name': chunk.metadata.get('source_file', 'Unknown'),
                'word_count': len(chunk.page_content.split())
            }

            chunk.page_content = f"{summary}\n{chunk.page_content}"
            enhanced_size = len(chunk.page_content)
            print(f"  - Summary generated in {chunk_time:.3f}s (size: {original_size} → {enhanced_size} chars)")

        summary_time = time.time() - summary_start
        print(f"All chunk summaries completed in {summary_time:.3f}s")

        # Create and save vector database
        print("Creating FAISS vector database...")
        vector_start = time.time()
        vectorstore = FAISS.from_documents(texts, embedding_model)
        vectorstore.save_local(faiss_db_path)
        vector_time = time.time() - vector_start
        print(f"FAISS database created and saved in {vector_time:.3f}s")

        total_create_time = time.time() - create_start
        print(f"Total database creation time: {total_create_time:.3f}s")

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    print(f"Retriever created with top_k={top_k}")
    print(f"--- CREATE_RETRIEVER END ---\n")
    return retriever

# Generate summaries for chunks using Ollama LLM
def generate_summary_with_ollama(chunk_text, surrounding_text, file_name):
    prompt = f"""
    You are given a text chunk from a document. Below is the chunk content, the surrounding chunks, and the file name.
    
    Chunk Content:
    {chunk_text}
    
    Surrounding Chunks:
    {surrounding_text}
    
    File Name:
    {file_name}
    
    Please provide a concise summary of the chunk content in the context of the surrounding chunks and the file name. Do not include any additional information or formatting.
    """
    
    summary = ollama_model_call(prompt)
    
    return summary.content

# Helper function to call Ollama model
def ollama_model_call(prompt):
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0.1,
        num_predict=1000,
        streaming=False
    )
    
    response = llm.invoke(prompt)
    
    return response