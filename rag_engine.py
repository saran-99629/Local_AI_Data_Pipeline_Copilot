import hashlib
from pathlib import Path

import chromadb
import requests


# Configuration
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Create a persistent ChromaDB database
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="spark_knowledge"
)


def get_embedding(text):
    """Convert text into a vector using Ollama."""

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def split_into_chunks(text, max_chars=800):
    """Split a document into small, meaningful chunks."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if (
            current_chunk
            and len(current_chunk) + len(paragraph) > max_chars
        ):
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk = (
                f"{current_chunk}\n\n{paragraph}"
                if current_chunk
                else paragraph
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def index_knowledge_base():
    """Read files, create embeddings, and store them."""

    files = list(KNOWLEDGE_DIR.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown files found in {KNOWLEDGE_DIR}"
        )

    documents = []
    ids = []
    metadatas = []
    embeddings = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{file_path.name}:{index}:{chunk}".encode("utf-8")
            ).hexdigest()

            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "source": file_path.name
            })
            embeddings.append(get_embedding(chunk))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


def retrieve_context(question, top_k=3):
    """Find the most relevant knowledge-base chunks."""

    if collection.count() == 0:
        index_knowledge_base()

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    context = "\n\n".join(
        f"Source: {metadata['source']}\n{chunk}"
        for chunk, metadata in zip(
            retrieved_chunks,
            retrieved_metadata
        )
    )

    sources = sorted({
        metadata["source"]
        for metadata in retrieved_metadata
    })

    return context, sources


if __name__ == "__main__":
    count = index_knowledge_base()
    print(f"Indexed {count} knowledge-base chunks.")

    question = (
        "How can I remove duplicate customer records "
        "and handle missing email values in PySpark?"
    )

    context, sources = retrieve_context(question)

    print("\n--- RETRIEVED CONTEXT ---\n")
    print(context)

    print("\n--- SOURCES ---")
    print(sources)