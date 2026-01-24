"""
Portable knowledge retrieval service for RAG.

Retrieves relevant knowledge from vector stores based on query similarity.
This module has no Django dependencies and can be used standalone.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from agent_runtime_core.vectorstore.base import VectorStore, VectorSearchResult
from agent_runtime_core.vectorstore.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for knowledge retrieval."""
    
    top_k: int = 5
    """Maximum number of chunks to retrieve."""
    
    similarity_threshold: float = 0.0
    """Minimum similarity score (0-1) to include in results."""
    
    include_metadata: bool = True
    """Whether to include metadata in results."""


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata and score."""
    
    content: str
    """The chunk text content."""
    
    score: float
    """Similarity score (higher = more similar)."""
    
    source_id: Optional[str] = None
    """ID of the source document."""
    
    source_name: Optional[str] = None
    """Human-readable name of the source."""
    
    chunk_index: Optional[int] = None
    """Index of this chunk within the source."""
    
    metadata: dict = field(default_factory=dict)
    """Additional metadata."""


class KnowledgeRetriever:
    """
    Portable service to retrieve relevant knowledge for RAG at runtime.
    
    Handles:
    - Embedding user queries
    - Searching vector store for similar content
    - Filtering by source and similarity threshold
    - Formatting retrieved content for inclusion in prompts
    
    This class has no Django dependencies and can be used in standalone
    Python scripts or any other context.
    
    Usage:
        from agent_runtime_core.vectorstore import get_vector_store, get_embedding_client
        from agent_runtime_core.rag import KnowledgeRetriever
        
        vector_store = get_vector_store("sqlite_vec", path="./vectors.db")
        embedding_client = get_embedding_client("openai")
        
        retriever = KnowledgeRetriever(vector_store, embedding_client)
        results = await retriever.retrieve(
            query="What is the return policy?",
            top_k=5,
        )
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        default_config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance for searching embeddings
            embedding_client: EmbeddingClient instance for embedding queries
            default_config: Default retrieval configuration
        """
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._default_config = default_config or RetrievalConfig()
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant knowledge chunks for a query.
        
        Args:
            query: The user's query to find relevant content for
            top_k: Maximum number of chunks to retrieve (overrides default)
            similarity_threshold: Minimum similarity score (overrides default)
            filter: Optional metadata filter for the search
            
        Returns:
            List of RetrievedChunk objects ordered by relevance
        """
        if top_k is None:
            top_k = self._default_config.top_k
        if similarity_threshold is None:
            similarity_threshold = self._default_config.similarity_threshold
        
        # Embed the query
        query_vector = await self._embedding_client.embed(query)
        
        # Search vector store
        results = await self._vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            filter=filter,
        )
        
        # Filter by similarity threshold and convert to RetrievedChunk
        retrieved = []
        for result in results:
            if result.score >= similarity_threshold:
                retrieved.append(RetrievedChunk(
                    content=result.content,
                    score=result.score,
                    source_id=result.metadata.get('source_id'),
                    source_name=result.metadata.get('name') or result.metadata.get('source_name'),
                    chunk_index=result.metadata.get('chunk_index'),
                    metadata=result.metadata if self._default_config.include_metadata else {},
                ))
        
        return retrieved
    
    async def retrieve_for_sources(
        self,
        query: str,
        source_ids: list[str],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks from specific sources.
        
        Args:
            query: The user's query
            source_ids: List of source IDs to search within
            top_k: Maximum number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of RetrievedChunk objects
        """
        # For now, we search all and filter
        # TODO: Support OR filters in vector stores for efficiency
        all_results = []

        for source_id in source_ids:
            results = await self.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filter={'source_id': source_id},
            )
            all_results.extend(results)

        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        if top_k:
            all_results = all_results[:top_k]

        return all_results

    async def retrieve_formatted(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter: Optional[dict] = None,
        header: str = "## Relevant Knowledge\n",
    ) -> str:
        """
        Retrieve and format knowledge for inclusion in a prompt.

        Args:
            query: The user's query
            top_k: Maximum number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            filter: Optional metadata filter
            header: Header text for the formatted output

        Returns:
            Formatted string of retrieved knowledge for prompt inclusion
        """
        results = await self.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter,
        )

        if not results:
            return ""

        return self.format_results(results, header=header)

    def format_results(
        self,
        results: list[RetrievedChunk],
        header: str = "## Relevant Knowledge\n",
    ) -> str:
        """
        Format retrieved results for inclusion in a prompt.

        Args:
            results: List of RetrievedChunk objects
            header: Header text for the formatted output

        Returns:
            Formatted string
        """
        if not results:
            return ""

        parts = [header]
        parts.append("The following information may be relevant to the user's question:\n")

        # Group by source
        by_source = {}
        for r in results:
            source = r.source_name or r.source_id or 'Unknown'
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(r)

        for source, chunks in by_source.items():
            parts.append(f"\n### {source}\n")
            for chunk in chunks:
                parts.append(f"{chunk.content}\n")

        return "\n".join(parts)

    async def close(self) -> None:
        """Close the retriever and release resources."""
        await self._vector_store.close()
        await self._embedding_client.close()

