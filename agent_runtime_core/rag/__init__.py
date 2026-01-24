"""
RAG (Retrieval Augmented Generation) module for agent_runtime_core.

This module provides portable RAG services that work without Django:
- KnowledgeIndexer: Service to chunk, embed, and store knowledge in vector stores
- KnowledgeRetriever: Service to retrieve relevant knowledge at runtime
- Text chunking utilities

Example usage:
    from agent_runtime_core.rag import (
        KnowledgeIndexer,
        KnowledgeRetriever,
        chunk_text,
        ChunkingConfig,
    )
    from agent_runtime_core.vectorstore import get_vector_store, get_embedding_client
    
    # Setup
    vector_store = get_vector_store("sqlite_vec", path="./vectors.db")
    embedding_client = get_embedding_client("openai")
    
    # Index content
    indexer = KnowledgeIndexer(vector_store, embedding_client)
    await indexer.index_text(
        text="Your knowledge content here...",
        source_id="doc-1",
        metadata={"name": "My Document"},
    )
    
    # Retrieve at runtime
    retriever = KnowledgeRetriever(vector_store, embedding_client)
    results = await retriever.retrieve(
        query="What is the return policy?",
        top_k=5,
    )
"""

from agent_runtime_core.rag.chunking import (
    chunk_text,
    ChunkingConfig,
    TextChunk,
)


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "KnowledgeIndexer":
        from agent_runtime_core.rag.indexer import KnowledgeIndexer
        return KnowledgeIndexer
    elif name == "KnowledgeRetriever":
        from agent_runtime_core.rag.retriever import KnowledgeRetriever
        return KnowledgeRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Chunking
    "chunk_text",
    "ChunkingConfig",
    "TextChunk",
    # Services
    "KnowledgeIndexer",
    "KnowledgeRetriever",
]

