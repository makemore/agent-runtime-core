"""
Portable knowledge indexing service for RAG.

Handles chunking, embedding, and storing knowledge in vector stores.
This module has no Django dependencies and can be used standalone.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol
from uuid import uuid4

from agent_runtime_core.rag.chunking import chunk_text, ChunkingConfig, TextChunk
from agent_runtime_core.vectorstore.base import VectorStore
from agent_runtime_core.vectorstore.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """Represents an indexed document with its metadata."""
    
    source_id: str
    """Unique identifier for the source document."""
    
    name: str = ""
    """Human-readable name for the document."""
    
    content_hash: str = ""
    """Hash of the content for change detection."""
    
    chunk_count: int = 0
    """Number of chunks created from this document."""
    
    indexed_at: Optional[datetime] = None
    """When the document was last indexed."""
    
    metadata: dict = field(default_factory=dict)
    """Additional metadata about the document."""


@dataclass
class IndexingResult:
    """Result of an indexing operation."""
    
    status: str
    """Status: 'success', 'skipped', or 'error'."""
    
    source_id: str
    """ID of the indexed source."""
    
    chunks_indexed: int = 0
    """Number of chunks that were indexed."""
    
    message: str = ""
    """Optional message with details."""
    
    error: Optional[str] = None
    """Error message if status is 'error'."""


def _compute_content_hash(content: str) -> str:
    """Compute a hash of the content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class KnowledgeIndexer:
    """
    Portable service to index knowledge sources for RAG retrieval.
    
    Handles:
    - Chunking text into appropriate sizes
    - Generating embeddings via configured provider
    - Storing vectors in the configured vector store
    
    This class has no Django dependencies and can be used in standalone
    Python scripts or any other context.
    
    Usage:
        from agent_runtime_core.vectorstore import get_vector_store, get_embedding_client
        from agent_runtime_core.rag import KnowledgeIndexer
        
        vector_store = get_vector_store("sqlite_vec", path="./vectors.db")
        embedding_client = get_embedding_client("openai")
        
        indexer = KnowledgeIndexer(vector_store, embedding_client)
        result = await indexer.index_text(
            text="Your document content...",
            source_id="doc-1",
            metadata={"name": "My Document"},
        )
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        default_chunking_config: Optional[ChunkingConfig] = None,
    ):
        """
        Initialize the indexer.
        
        Args:
            vector_store: VectorStore instance for storing embeddings
            embedding_client: EmbeddingClient instance for generating embeddings
            default_chunking_config: Default chunking configuration
        """
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._default_chunking_config = default_chunking_config or ChunkingConfig()
    
    async def index_text(
        self,
        text: str,
        source_id: str,
        metadata: Optional[dict] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        force: bool = False,
        content_hash: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index text content for RAG retrieval.
        
        Args:
            text: The text content to index
            source_id: Unique identifier for this content source
            metadata: Optional metadata to store with each chunk
            chunking_config: Optional chunking configuration override
            force: If True, re-index even if content hasn't changed
            content_hash: Optional pre-computed content hash for change detection
            
        Returns:
            IndexingResult with status and details
        """
        if metadata is None:
            metadata = {}
        
        if chunking_config is None:
            chunking_config = self._default_chunking_config
        
        # Compute content hash for change detection
        computed_hash = content_hash or _compute_content_hash(text)
        
        try:
            if not text or not text.strip():
                return IndexingResult(
                    status='error',
                    source_id=source_id,
                    error='No content to index',
                )
            
            # Chunk the content
            chunks = chunk_text(
                text,
                config=chunking_config,
                metadata={
                    'source_id': source_id,
                    **metadata,
                },
            )
            
            if not chunks:
                return IndexingResult(
                    status='error',
                    source_id=source_id,
                    error='Content produced no chunks',
                )

            # Delete existing vectors for this source
            await self._delete_existing_vectors(source_id)

            # Generate embeddings and store
            chunk_ids = await self._embed_and_store_chunks(source_id, chunks, metadata)

            return IndexingResult(
                status='success',
                source_id=source_id,
                chunks_indexed=len(chunk_ids),
            )

        except Exception as e:
            logger.exception(f"Error indexing source {source_id}")
            return IndexingResult(
                status='error',
                source_id=source_id,
                error=str(e),
            )

    async def _delete_existing_vectors(self, source_id: str) -> int:
        """Delete existing vectors for a source."""
        try:
            deleted = await self._vector_store.delete_by_filter({
                'source_id': str(source_id),
            })
            logger.debug(f"Deleted {deleted} existing vectors for source {source_id}")
            return deleted
        except Exception as e:
            logger.warning(f"Error deleting existing vectors: {e}")
            return 0

    async def _embed_and_store_chunks(
        self,
        source_id: str,
        chunks: list[TextChunk],
        metadata: dict,
    ) -> list[str]:
        """Generate embeddings and store chunks in vector store."""
        chunk_ids = []

        # Batch embed for efficiency
        texts = [chunk.text for chunk in chunks]
        embeddings = await self._embedding_client.embed_batch(texts)

        # Store each chunk
        items = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{source_id}_{chunk.index}"
            items.append((
                chunk_id,
                embedding,
                chunk.text,
                {
                    'source_id': str(source_id),
                    'chunk_index': chunk.index,
                    'total_chunks': len(chunks),
                    **metadata,
                },
            ))
            chunk_ids.append(chunk_id)

        await self._vector_store.add_batch(items)
        return chunk_ids

    async def delete_source(self, source_id: str) -> int:
        """
        Delete all vectors for a source.

        Args:
            source_id: The source ID to delete vectors for

        Returns:
            Number of vectors deleted
        """
        return await self._delete_existing_vectors(source_id)

    async def close(self) -> None:
        """Close the indexer and release resources."""
        await self._vector_store.close()
        await self._embedding_client.close()

