"""
Vector store module for agent_runtime_core.

Provides pluggable vector storage backends for similarity search:
- sqlite-vec: Lightweight, local development (requires: pip install sqlite-vec)
- pgvector: Production PostgreSQL (requires: pip install pgvector psycopg[binary])
- Vertex AI: Enterprise-scale managed (requires: pip install google-cloud-aiplatform)

Example usage:
    from agent_runtime_core.vectorstore import (
        get_vector_store,
        get_embedding_client,
        VectorStore,
        EmbeddingClient,
    )

    # Get clients
    vector_store = get_vector_store("sqlite_vec", path="./vectors.db")
    embeddings = get_embedding_client("openai")

    # Index a document
    text = "The quick brown fox jumps over the lazy dog"
    vector = await embeddings.embed(text)
    await vector_store.add(
        id="doc-1",
        vector=vector,
        content=text,
        metadata={"source": "example"},
    )

    # Search
    query = "fast animal"
    query_vector = await embeddings.embed(query)
    results = await vector_store.search(query_vector, limit=5)
    for r in results:
        print(f"{r.score:.3f}: {r.content}")
"""

from typing import Optional

from agent_runtime_core.vectorstore.base import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
)
from agent_runtime_core.vectorstore.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddings,
    VertexAIEmbeddings,
)


def get_vector_store(backend: str = "sqlite_vec", **kwargs) -> VectorStore:
    """
    Factory function to get a vector store instance.

    Args:
        backend: The backend to use. Options:
            - "sqlite_vec": SQLite with sqlite-vec extension (default)
            - "pgvector": PostgreSQL with pgvector extension
            - "vertex": Google Vertex AI Vector Search
        **kwargs: Backend-specific configuration options

    Returns:
        A VectorStore instance

    Raises:
        ValueError: If the backend is unknown
        ImportError: If required dependencies are not installed

    Examples:
        # SQLite-vec (local development)
        store = get_vector_store("sqlite_vec", path="./vectors.db")

        # PGVector (production PostgreSQL)
        store = get_vector_store("pgvector", connection_string="postgresql://...")

        # Vertex AI (enterprise scale)
        store = get_vector_store(
            "vertex",
            project_id="my-project",
            location="us-central1",
            index_endpoint_id="...",
            deployed_index_id="...",
        )
    """
    if backend == "sqlite_vec":
        from agent_runtime_core.vectorstore.sqlite_vec import SqliteVecStore

        return SqliteVecStore(**kwargs)
    elif backend == "pgvector":
        # Import from django_agent_runtime if available
        try:
            from django_agent_runtime.vectorstore import PgVectorStore

            return PgVectorStore(**kwargs)
        except ImportError:
            raise ImportError(
                "PgVectorStore requires django_agent_runtime. "
                "Install with: pip install django-agent-runtime[pgvector]"
            )
    elif backend == "vertex":
        from agent_runtime_core.vectorstore.vertex import VertexVectorStore

        return VertexVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unknown vector store backend: {backend}. "
            f"Available backends: sqlite_vec, pgvector, vertex"
        )


def get_embedding_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> EmbeddingClient:
    """
    Factory function to get an embedding client.

    Args:
        provider: The provider to use. Options:
            - "openai": OpenAI embeddings (default)
            - "vertex": Google Vertex AI embeddings
        model: Optional model name override
        **kwargs: Provider-specific configuration options

    Returns:
        An EmbeddingClient instance

    Raises:
        ValueError: If the provider is unknown
        ImportError: If required dependencies are not installed

    Examples:
        # OpenAI (default model: text-embedding-3-small)
        client = get_embedding_client("openai")

        # OpenAI with specific model
        client = get_embedding_client("openai", model="text-embedding-3-large")

        # Vertex AI
        client = get_embedding_client(
            "vertex",
            model="text-embedding-004",
            project_id="my-project",
        )
    """
    if provider == "openai":
        if model:
            kwargs["model"] = model
        return OpenAIEmbeddings(**kwargs)
    elif provider == "vertex":
        if model:
            kwargs["model"] = model
        return VertexAIEmbeddings(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Available providers: openai, vertex"
        )


# Lazy imports for optional backends
def __getattr__(name: str):
    """Lazy import for optional backends."""
    if name == "SqliteVecStore":
        from agent_runtime_core.vectorstore.sqlite_vec import SqliteVecStore

        return SqliteVecStore
    elif name == "VertexVectorStore":
        from agent_runtime_core.vectorstore.vertex import VertexVectorStore

        return VertexVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Abstract interfaces
    "VectorStore",
    "VectorRecord",
    "VectorSearchResult",
    "EmbeddingClient",
    # Implementations
    "OpenAIEmbeddings",
    "VertexAIEmbeddings",
    "SqliteVecStore",
    "VertexVectorStore",
    # Factory functions
    "get_vector_store",
    "get_embedding_client",
]

