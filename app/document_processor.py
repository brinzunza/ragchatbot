import os
import glob
import threading
import time
from datetime import datetime
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from llm_config import get_llm

_COLLECTION_NAME = "documents"

# Guards the "check what's already indexed, then add only what's new" sequence
# so two uploads racing each other don't both decide the same file is new and
# redundantly re-summarize it. Chroma's own writes are safe to interleave
# (each add_documents call is a real incremental insert against the shared
# on-disk SQLite store, not a load-mutate-save-the-whole-thing cycle), so this
# lock is about avoiding wasted duplicate work, not about preventing corruption.
_index_lock = threading.Lock()

# Serializes every call into the local embedding model. NomicEmbeddings
# (inference_mode="local") runs on gpt4all's llama.cpp backend, which is not
# safe to call from two threads at once - observed firsthand: a chat query's
# embed_query racing an upload's embed_documents crashed the entire process
# with `GGML_ASSERT: backend_embd != nullptr`, not just raised a Python
# exception. Every embedding call in this app - indexing or retrieval - goes
# through _LockedEmbeddings so that's no longer possible regardless of which
# code path (upload, chat, LangGraph retriever node) triggers it.
_embedding_lock = threading.Lock()


class _LockedEmbeddings(Embeddings):
    """Wraps an embeddings model so embed calls are serialized process-wide."""

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts):
        with _embedding_lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text):
        with _embedding_lock:
            return self._inner.embed_query(text)


def _get_vectorstore(chroma_db_path: str) -> Chroma:
    """
    Get a Chroma vectorstore handle. Cheap to call repeatedly - a fresh
    PersistentClient pointed at the same path sees whatever the most recent
    writer persisted, confirmed empirically, so there's no need to cache and
    invalidate a shared client instance across calls.

    Args:
        chroma_db_path: Path where the Chroma database is stored

    Returns:
        Chroma vectorstore instance
    """
    embedding_model = _LockedEmbeddings(
        NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    )
    client = chromadb.PersistentClient(path=chroma_db_path)
    return Chroma(client=client, collection_name=_COLLECTION_NAME, embedding_function=embedding_model)


def has_documents(chroma_db_path: str) -> bool:
    """
    Whether any documents are currently indexed.

    Args:
        chroma_db_path: Path where the Chroma database is stored

    Returns:
        True if at least one chunk is indexed
    """
    vectorstore = _get_vectorstore(chroma_db_path)
    result = vectorstore.get(include=[])
    return len(result["ids"]) > 0


def list_documents(chroma_db_path: str, files_path: str) -> List[dict]:
    """
    List indexed documents with their chunk counts and upload dates.

    Args:
        chroma_db_path: Path where the Chroma database is stored
        files_path: Path to the folder containing the cached markdown files

    Returns:
        List of dicts with 'name', 'chunk_count', and 'uploaded_at' keys
    """
    vectorstore = _get_vectorstore(chroma_db_path)
    result = vectorstore.get(include=["metadatas"])

    chunk_counts: dict = {}
    for metadata in result["metadatas"]:
        source = metadata.get('file_name', 'Unknown')
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


def delete_document(name: str, chroma_db_path: str, files_path: str) -> bool:
    """
    Delete a single document's chunks from the Chroma collection and remove
    its cached markdown file. Unlike FAISS, an empty collection is a valid
    persisted state, so there's no special-case cleanup needed when the last
    document is removed.

    Args:
        name: Display name of the document (matches what list_documents returns)
        chroma_db_path: Path where the Chroma database is stored
        files_path: Path to the folder containing the cached markdown files

    Returns:
        True if the document was found and deleted, False if it wasn't found
    """
    target_path = os.path.join(files_path, f"{name}.md")
    vectorstore = _get_vectorstore(chroma_db_path)

    with _index_lock:
        existing = vectorstore.get(where={"file_name": target_path}, include=[])
        ids_to_delete = existing["ids"]

        if not ids_to_delete:
            return False

        vectorstore.delete(ids=ids_to_delete)

    if os.path.exists(target_path):
        os.remove(target_path)

    return True


def get_retriever(chroma_db_path: str = "chroma_db", top_k: int = 3):
    """
    Get a retriever over the existing Chroma collection for querying.

    Args:
        chroma_db_path: Path where the Chroma database is stored
        top_k: Number of top documents to retrieve

    Returns:
        Configured retriever instance

    Raises:
        ValueError: if no documents have been indexed yet
    """
    if not has_documents(chroma_db_path):
        raise ValueError(f"No documents found at {chroma_db_path}")

    vectorstore = _get_vectorstore(chroma_db_path)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def index_documents(
    md_folder_path: str,
    chroma_db_path: str = "chroma_db",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    progress_callback=None,
):
    """
    Ensure every .md file in md_folder_path is embedded in the Chroma
    collection, adding only whichever files aren't indexed yet.

    The slow part - chunking and per-chunk AI summarization - runs without
    holding any lock, so concurrent chat queries (get_retriever) aren't
    blocked for the whole indexing duration.

    Args:
        md_folder_path: Path to the folder containing markdown files
        chroma_db_path: Path where the Chroma database will be stored
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        progress_callback: optional callable(current_chunk, total_chunks, file_name)
            invoked after each chunk is summarized, for progress reporting
    """
    print(f"\n--- INDEX_DOCUMENTS START ---")
    print(f"MD folder path: {md_folder_path}")
    print(f"Chroma DB path: {chroma_db_path}")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    md_files = glob.glob(os.path.join(md_folder_path, "*.md"))
    print(f"Found {len(md_files)} markdown files: {[os.path.basename(f) for f in md_files]}")

    if not md_files:
        raise ValueError(f"No .md files found in {md_folder_path}")

    vectorstore = _get_vectorstore(chroma_db_path)

    with _index_lock:
        result = vectorstore.get(include=["metadatas"])
        indexed_sources = {m.get('file_name') for m in result["metadatas"]}

    new_files = [f for f in md_files if f not in indexed_sources]

    if not new_files:
        print("No new files to index")
        print(f"--- INDEX_DOCUMENTS END ---\n")
        return

    print(f"Indexing {len(new_files)} new file(s): {[os.path.basename(f) for f in new_files]}")
    texts = _process_markdown_files(new_files, chunk_size, chunk_overlap, progress_callback)

    if not texts:
        raise ValueError("No documents were successfully loaded")

    with _index_lock:
        vectorstore.add_documents(texts)
    print(f"Added {len(texts)} chunks from {len(new_files)} new file(s)")

    print(f"--- INDEX_DOCUMENTS END ---\n")


def _process_markdown_files(md_files: List[str], chunk_size: int, chunk_overlap: int, progress_callback=None) -> List:
    """
    Load, chunk, and contextually summarize a set of markdown files.

    Args:
        md_files: Paths to the .md files to process
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        progress_callback: optional callable(current_chunk, total_chunks, file_name)
            invoked after each chunk is summarized

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

        if progress_callback:
            progress_callback(i + 1, amount, os.path.basename(chunk.metadata['file_name']))

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
