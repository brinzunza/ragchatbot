import os
import glob
import shutil
import time
from datetime import datetime
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from llm_config import get_llm


def _load_vectorstore(faiss_db_path: str) -> FAISS:
    """
    Load an existing FAISS vector database from disk.

    Args:
        faiss_db_path: Path where the FAISS database is stored

    Returns:
        Loaded FAISS vectorstore instance
    """
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    return FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)


def list_documents(faiss_db_path: str, files_path: str) -> List[dict]:
    """
    List indexed documents with their chunk counts and upload dates.

    Args:
        faiss_db_path: Path where the FAISS database is stored
        files_path: Path to the folder containing the cached markdown files

    Returns:
        List of dicts with 'name', 'chunk_count', and 'uploaded_at' keys
    """
    if not os.path.exists(faiss_db_path):
        return []

    vectorstore = _load_vectorstore(faiss_db_path)

    chunk_counts: dict = {}
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get('file_name', 'Unknown')
        chunk_counts[source] = chunk_counts.get(source, 0) + 1

    documents = []
    for source, count in chunk_counts.items():
        name = os.path.basename(source)
        if name.endswith('.md'):
            name = name[:-3]
        uploaded_at = ""
        if os.path.exists(source):
            uploaded_at = datetime.fromtimestamp(os.path.getmtime(source)).isoformat()
        documents.append({
            'name': name,
            'chunk_count': count,
            'uploaded_at': uploaded_at
        })

    documents.sort(key=lambda d: d['name'].lower())
    return documents


def delete_document(name: str, faiss_db_path: str, files_path: str) -> bool:
    """
    Delete a single document's chunks from the FAISS index without rebuilding
    the rest of the index, and remove its cached markdown file.

    Args:
        name: Display name of the document (matches what list_documents returns)
        faiss_db_path: Path where the FAISS database is stored
        files_path: Path to the folder containing the cached markdown files

    Returns:
        True if the document was found and deleted, False if it wasn't found
    """
    if not os.path.exists(faiss_db_path):
        return False

    target_path = os.path.join(files_path, f"{name}.md")
    vectorstore = _load_vectorstore(faiss_db_path)

    ids_to_delete = [
        doc_id for doc_id, doc in vectorstore.docstore._dict.items()
        if doc.metadata.get('file_name') == target_path
    ]

    if not ids_to_delete:
        return False

    vectorstore.delete(ids_to_delete)

    if os.path.exists(target_path):
        os.remove(target_path)

    if len(vectorstore.index_to_docstore_id) == 0:
        # No documents left - remove the index entirely rather than persist an
        # empty one, so /database/status correctly reports exists=False.
        shutil.rmtree(faiss_db_path)
    else:
        vectorstore.save_local(faiss_db_path)

    return True


def create_retriever(
    md_folder_path: str,
    faiss_db_path: str = "faiss_db",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    top_k: int = 3
):
    """
    Create a document retriever from markdown files.

    This function processes markdown files, splits them into chunks, enhances them with
    AI-generated summaries, and creates a FAISS vector database for semantic search.

    Args:
        md_folder_path: Path to the folder containing markdown files
        faiss_db_path: Path where the FAISS database will be stored
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        top_k: Number of top documents to retrieve

    Returns:
        Configured retriever instance
    """
    print(f"\n--- CREATE_RETRIEVER START ---")
    print(f"MD folder path: {md_folder_path}")
    print(f"FAISS DB path: {faiss_db_path}")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}, top_k: {top_k}")

    md_files = glob.glob(os.path.join(md_folder_path, "*.md"))
    print(f"Found {len(md_files)} markdown files: {[os.path.basename(f) for f in md_files]}")

    if os.path.exists(faiss_db_path):
        # Load the existing index, then find any .md files on disk that
        # aren't in it yet (e.g. from a later upload) and add just those -
        # reloading here would silently ignore new files and only reprocessing
        # everything would redo (expensive) summarization on unchanged docs.
        print(f"Loading existing FAISS database from {faiss_db_path}")
        load_start = time.time()
        vectorstore = _load_vectorstore(faiss_db_path)
        load_time = time.time() - load_start
        print(f"FAISS database loaded in {load_time:.3f}s")

        indexed_sources = {
            doc.metadata.get('file_name')
            for doc in vectorstore.docstore._dict.values()
        }
        new_files = [f for f in md_files if f not in indexed_sources]

        if new_files:
            print(f"Indexing {len(new_files)} new file(s): {[os.path.basename(f) for f in new_files]}")
            texts = _process_markdown_files(new_files, chunk_size, chunk_overlap)
            if texts:
                vectorstore.add_documents(texts)
                vectorstore.save_local(faiss_db_path)
                print(f"Added {len(texts)} chunks from {len(new_files)} new file(s)")
        else:
            print("No new files to index")
    else:
        if not md_files:
            raise ValueError(f"No .md files found in {md_folder_path}")

        print("Creating new FAISS database...")
        create_start = time.time()

        embedding_start = time.time()
        embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        embedding_init_time = time.time() - embedding_start
        print(f"Embedding model initialized in {embedding_init_time:.3f}s")

        texts = _process_markdown_files(md_files, chunk_size, chunk_overlap)
        if not texts:
            raise ValueError("No documents were successfully loaded")

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


def _process_markdown_files(md_files: List[str], chunk_size: int, chunk_overlap: int) -> List:
    """
    Load, chunk, and contextually summarize a set of markdown files.

    Args:
        md_files: Paths to the .md files to process
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of chunked, summary-enhanced Document objects ready to embed
    """
    all_documents = []

    file_load_start = time.time()
    for i, md_file in enumerate(md_files):
        print(f"Processing file {i+1}/{len(md_files)}: {os.path.basename(md_file)}")
        try:
            loader = TextLoader(md_file, encoding='utf-8')
            documents = loader.load()

            for doc in documents:
                doc_size = len(doc.page_content)
                doc.metadata.update({
                    'file_name': md_file,
                    'source_file': md_file
                })
                print(f"  - Document loaded: {doc_size} characters")

            all_documents.extend(documents)
            print(f"  - Completed: {os.path.basename(md_file)} ({len(documents)} document(s))")

        except Exception as e:
            print(f"  ERROR loading {md_file}: {e}")
            continue

    file_load_time = time.time() - file_load_start
    print(f"All files loaded in {file_load_time:.3f}s (total docs: {len(all_documents)})")

    if not all_documents:
        return []

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

    print("Enhancing chunks with AI-generated summaries...")
    summary_start = time.time()
    for i, chunk in enumerate(texts):
        surrounding_chunks = texts[max(0, i-1):i] + texts[i+1:i+2]
        surrounding_text = "\n\n".join([c.page_content for c in surrounding_chunks])

        amount = len(texts)
        chunk_start = time.time()
        print(f"Processing chunk {i+1}/{amount} from {os.path.basename(chunk.metadata['source_file'])}")

        summary = _generate_summary_with_ollama(
            chunk.page_content,
            surrounding_text,
            chunk.metadata['source_file']
        )
        chunk_time = time.time() - chunk_start

        original_size = len(chunk.page_content)
        chunk.metadata = {
            'file_name': chunk.metadata.get('source_file', 'Unknown'),
            'word_count': len(chunk.page_content.split()),
            # Preserved so the frontend can highlight the exact source span -
            # start_index comes from RecursiveCharacterTextSplitter(add_start_index=True)
            # and refers to the original (pre-summary) chunk text below.
            'start_index': chunk.metadata.get('start_index', 0),
            'chunk_length': original_size,
        }

        chunk.page_content = f"{summary}\n{chunk.page_content}"
        enhanced_size = len(chunk.page_content)
        print(f"  - Summary generated in {chunk_time:.3f}s (size: {original_size} → {enhanced_size} chars)")

    summary_time = time.time() - summary_start
    print(f"All chunk summaries completed in {summary_time:.3f}s")

    return texts


def _generate_summary_with_ollama(chunk_text, surrounding_text, file_name):
    """
    Generate a summary for a text chunk using configured LLM (OpenAI or Ollama).

    Args:
        chunk_text: The text chunk to summarize
        surrounding_text: Text from surrounding chunks for context
        file_name: Name of the source file

    Returns:
        Generated summary string
    """
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

    llm = get_llm(
        temperature=0.1,
        streaming=False,
        num_predict=1000  # Only used by Ollama
    )

    response = llm.invoke(prompt)
    return response.content
