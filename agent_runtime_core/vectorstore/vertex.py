"""
Vertex AI Vector Search implementation.

Uses Google Cloud Vertex AI Vector Search for enterprise-scale vector similarity search.
Ideal for production deployments requiring managed infrastructure and high scalability.

Requires: pip install google-cloud-aiplatform

Note: Vertex AI Vector Search requires pre-created index and endpoint resources.
See: https://cloud.google.com/vertex-ai/docs/vector-search/overview
"""

import json
from typing import Optional

from agent_runtime_core.vectorstore.base import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
)


class VertexVectorStore(VectorStore):
    """
    Vector store using Google Vertex AI Vector Search.

    This implementation uses Vertex AI's managed vector search service for
    enterprise-scale similarity search. It's ideal for:
    - Production deployments requiring high availability
    - Large-scale datasets (billions of vectors)
    - Applications already using Google Cloud

    Important: This store requires pre-created Vertex AI resources:
    1. A Vector Search Index
    2. An Index Endpoint with the index deployed

    The store maintains a local cache of content/metadata since Vertex AI
    Vector Search only stores vectors and IDs. For production use, consider
    using a separate database for content storage.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_id: str,
        deployed_index_id: str,
        index_id: Optional[str] = None,
    ):
        """
        Initialize Vertex AI Vector Search store.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (e.g., "us-central1")
            index_endpoint_id: The ID of the index endpoint
            deployed_index_id: The ID of the deployed index
            index_id: Optional index ID for batch updates
        """
        self._project_id = project_id
        self._location = location
        self._index_endpoint_id = index_endpoint_id
        self._deployed_index_id = deployed_index_id
        self._index_id = index_id
        self._endpoint = None
        self._initialized = False

        # Local cache for content and metadata
        # In production, use a proper database
        self._content_cache: dict[str, tuple[str, dict]] = {}

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK and get endpoint."""
        if self._initialized:
            return
        try:
            from google.cloud import aiplatform
            from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
        except ImportError:
            raise ImportError(
                "Google Cloud AI Platform package not installed. "
                "Install with: pip install google-cloud-aiplatform"
            )

        aiplatform.init(project=self._project_id, location=self._location)

        # Get the index endpoint
        endpoint_resource_name = (
            f"projects/{self._project_id}/locations/{self._location}/"
            f"indexEndpoints/{self._index_endpoint_id}"
        )
        self._endpoint = MatchingEngineIndexEndpoint(endpoint_resource_name)
        self._initialized = True

    async def add(
        self,
        id: str,
        vector: list[float],
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a vector with its content and metadata.

        Note: For Vertex AI, vectors are typically added through batch operations
        to the index. This method adds to a local cache and should be followed
        by a batch update to the index.
        """
        self._ensure_initialized()

        # Store content and metadata locally
        self._content_cache[id] = (content, metadata or {})

        # For single adds, we use the streaming update API if available
        # This requires the index to support streaming updates
        if self._index_id:
            import asyncio
            from google.cloud import aiplatform

            index_resource_name = (
                f"projects/{self._project_id}/locations/{self._location}/"
                f"indexes/{self._index_id}"
            )
            index = aiplatform.MatchingEngineIndex(index_resource_name)

            # Prepare datapoint
            datapoint = {
                "datapoint_id": id,
                "feature_vector": vector,
            }

            # Run in executor since SDK is synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: index.upsert_datapoints(datapoints=[datapoint]),
            )

    async def add_batch(
        self,
        items: list[tuple[str, list[float], str, Optional[dict]]],
    ) -> None:
        """Add multiple vectors efficiently."""
        if not items:
            return

        self._ensure_initialized()

        # Store content and metadata locally
        for id, _, content, metadata in items:
            self._content_cache[id] = (content, metadata or {})

        if self._index_id:
            import asyncio
            from google.cloud import aiplatform

            index_resource_name = (
                f"projects/{self._project_id}/locations/{self._location}/"
                f"indexes/{self._index_id}"
            )
            index = aiplatform.MatchingEngineIndex(index_resource_name)

            # Prepare datapoints
            datapoints = [
                {"datapoint_id": id, "feature_vector": vector}
                for id, vector, _, _ in items
            ]

            # Run in executor since SDK is synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: index.upsert_datapoints(datapoints=datapoints),
            )

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        self._ensure_initialized()
        import asyncio

        # Vertex AI Vector Search uses find_neighbors for queries
        loop = asyncio.get_event_loop()

        # Build the query
        queries = [query_vector]

        # Execute search
        response = await loop.run_in_executor(
            None,
            lambda: self._endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=queries,
                num_neighbors=limit,
            ),
        )

        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                id = neighbor.id
                # Vertex AI returns distance, convert to similarity
                # Using cosine similarity: score = 1 - distance for normalized vectors
                score = 1.0 - neighbor.distance if hasattr(neighbor, "distance") else 0.0

                # Get content and metadata from cache
                content, metadata = self._content_cache.get(id, ("", {}))

                # Apply filter if provided
                if filter:
                    match = all(
                        metadata.get(k) == v for k, v in filter.items()
                    )
                    if not match:
                        continue

                results.append(
                    VectorSearchResult(
                        id=id,
                        content=content,
                        score=score,
                        metadata=metadata,
                    )
                )

        return results

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        self._ensure_initialized()

        # Remove from local cache
        existed = id in self._content_cache
        self._content_cache.pop(id, None)

        # Delete from index if index_id is provided
        if self._index_id:
            import asyncio
            from google.cloud import aiplatform

            index_resource_name = (
                f"projects/{self._project_id}/locations/{self._location}/"
                f"indexes/{self._index_id}"
            )
            index = aiplatform.MatchingEngineIndex(index_resource_name)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: index.remove_datapoints(datapoint_ids=[id]),
            )

        return existed

    async def delete_by_filter(self, filter: dict) -> int:
        """Delete vectors matching filter."""
        # Find matching IDs from cache
        ids_to_delete = []
        for id, (_, metadata) in self._content_cache.items():
            if all(metadata.get(k) == v for k, v in filter.items()):
                ids_to_delete.append(id)

        # Delete each matching ID
        for id in ids_to_delete:
            await self.delete(id)

        return len(ids_to_delete)

    async def get(self, id: str) -> Optional[VectorRecord]:
        """
        Get a vector by ID.

        Note: Vertex AI Vector Search doesn't support direct vector retrieval.
        This returns cached content/metadata with an empty vector.
        """
        if id not in self._content_cache:
            return None

        content, metadata = self._content_cache[id]
        return VectorRecord(
            id=id,
            vector=[],  # Vertex AI doesn't support vector retrieval
            content=content,
            metadata=metadata,
        )

    async def close(self) -> None:
        """Close connections."""
        self._endpoint = None
        self._initialized = False

