"""
Abstract base classes for vector storage backends.

These interfaces define the contract that all vector storage backends must implement.
Implementations can use different backends like sqlite-vec, pgvector, or Vertex AI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorRecord:
    """A stored vector with its content and metadata."""

    id: str
    vector: list[float]
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """A search result with similarity score."""

    id: str
    content: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """
    Abstract interface for vector storage and similarity search.

    Vector stores handle storing embeddings and performing similarity searches.
    Different backends can be used depending on scale and deployment requirements:
    - sqlite-vec: Lightweight, local development
    - pgvector: Production PostgreSQL deployments
    - Vertex AI: Enterprise-scale managed infrastructure
    """

    @abstractmethod
    async def add(
        self,
        id: str,
        vector: list[float],
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a vector with its content and metadata.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector
            content: Original text content
            metadata: Optional metadata dictionary
        """
        ...

    @abstractmethod
    async def add_batch(
        self,
        items: list[tuple[str, list[float], str, Optional[dict]]],
    ) -> None:
        """
        Add multiple vectors efficiently.

        Args:
            items: List of (id, vector, content, metadata) tuples
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filter: Optional metadata filter (equality matching)

        Returns:
            List of VectorSearchResult ordered by similarity (highest first)
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
            id: The vector ID to delete

        Returns:
            True if the vector existed and was deleted
        """
        ...

    @abstractmethod
    async def delete_by_filter(self, filter: dict) -> int:
        """
        Delete vectors matching filter.

        Args:
            filter: Metadata filter for deletion (equality matching)

        Returns:
            Number of vectors deleted
        """
        ...

    @abstractmethod
    async def get(self, id: str) -> Optional[VectorRecord]:
        """
        Get a vector by ID.

        Args:
            id: The vector ID to retrieve

        Returns:
            VectorRecord if found, None otherwise
        """
        ...

    async def close(self) -> None:
        """Close connections. Override if needed."""
        pass

