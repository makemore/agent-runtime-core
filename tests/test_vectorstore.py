"""
Tests for vector store implementations.

These tests verify the VectorStore interface and SqliteVecStore implementation.
"""

import pytest
from uuid import uuid4


class TestVectorStoreInterface:
    """Tests for the VectorStore abstract interface."""

    def test_vector_record_dataclass(self):
        """Test VectorRecord dataclass."""
        from agent_runtime_core.vectorstore import VectorRecord

        record = VectorRecord(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            content="Hello world",
            metadata={"source": "test"},
        )
        assert record.id == "test-1"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.content == "Hello world"
        assert record.metadata == {"source": "test"}

    def test_vector_search_result_dataclass(self):
        """Test VectorSearchResult dataclass."""
        from agent_runtime_core.vectorstore import VectorSearchResult

        result = VectorSearchResult(
            id="test-1",
            content="Hello world",
            score=0.95,
            metadata={"source": "test"},
        )
        assert result.id == "test-1"
        assert result.content == "Hello world"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}


class TestSqliteVecStore:
    """Tests for SqliteVecStore implementation."""

    @pytest.fixture
    def store(self):
        """Create an in-memory SqliteVecStore for testing."""
        pytest.importorskip("sqlite_vec")
        from agent_runtime_core.vectorstore import SqliteVecStore

        return SqliteVecStore(":memory:")

    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        """Test adding and retrieving a vector."""
        await store.add(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            content="Hello world",
            metadata={"source": "test"},
        )

        record = await store.get("test-1")
        assert record is not None
        assert record.id == "test-1"
        assert record.content == "Hello world"
        assert record.metadata == {"source": "test"}
        # Vector should be approximately equal (floating point)
        assert len(record.vector) == 3
        assert abs(record.vector[0] - 0.1) < 0.001

    @pytest.mark.asyncio
    async def test_add_batch(self, store):
        """Test batch adding vectors."""
        items = [
            ("test-1", [0.1, 0.2, 0.3], "First", {"idx": 1}),
            ("test-2", [0.4, 0.5, 0.6], "Second", {"idx": 2}),
            ("test-3", [0.7, 0.8, 0.9], "Third", {"idx": 3}),
        ]
        await store.add_batch(items)

        for id, _, content, metadata in items:
            record = await store.get(id)
            assert record is not None
            assert record.content == content

    @pytest.mark.asyncio
    async def test_search(self, store):
        """Test similarity search."""
        # Add some vectors
        await store.add("doc-1", [1.0, 0.0, 0.0], "Document about cats")
        await store.add("doc-2", [0.0, 1.0, 0.0], "Document about dogs")
        await store.add("doc-3", [0.9, 0.1, 0.0], "Document about kittens")

        # Search for vectors similar to [1.0, 0.0, 0.0]
        results = await store.search([1.0, 0.0, 0.0], limit=2)

        assert len(results) == 2
        # First result should be doc-1 (exact match)
        assert results[0].id == "doc-1"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store):
        """Test similarity search with metadata filter."""
        await store.add("doc-1", [1.0, 0.0, 0.0], "Cat doc", {"type": "animal"})
        await store.add("doc-2", [0.9, 0.1, 0.0], "Kitten doc", {"type": "animal"})
        await store.add("doc-3", [0.8, 0.2, 0.0], "Car doc", {"type": "vehicle"})

        # Search with filter
        results = await store.search(
            [1.0, 0.0, 0.0],
            limit=10,
            filter={"type": "animal"},
        )

        # Should only return animal documents
        assert all(r.metadata.get("type") == "animal" for r in results)

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting a vector."""
        await store.add("test-1", [0.1, 0.2, 0.3], "Hello")

        # Verify it exists
        assert await store.get("test-1") is not None

        # Delete it
        deleted = await store.delete("test-1")
        assert deleted is True

        # Verify it's gone
        assert await store.get("test-1") is None

        # Delete non-existent
        deleted = await store.delete("non-existent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_by_filter(self, store):
        """Test deleting vectors by filter."""
        await store.add("doc-1", [1.0, 0.0, 0.0], "Cat", {"type": "animal"})
        await store.add("doc-2", [0.0, 1.0, 0.0], "Dog", {"type": "animal"})
        await store.add("doc-3", [0.0, 0.0, 1.0], "Car", {"type": "vehicle"})

        # Delete all animals
        deleted = await store.delete_by_filter({"type": "animal"})
        assert deleted == 2

        # Verify only vehicle remains
        assert await store.get("doc-1") is None
        assert await store.get("doc-2") is None
        assert await store.get("doc-3") is not None

    @pytest.mark.asyncio
    async def test_update_existing(self, store):
        """Test updating an existing vector."""
        await store.add("test-1", [0.1, 0.2, 0.3], "Original")

        # Update with same ID
        await store.add("test-1", [0.4, 0.5, 0.6], "Updated", {"new": True})

        record = await store.get("test-1")
        assert record.content == "Updated"
        assert record.metadata == {"new": True}

    @pytest.mark.asyncio
    async def test_close(self, store):
        """Test closing the store."""
        await store.add("test-1", [0.1, 0.2, 0.3], "Hello")
        await store.close()
        # After close, operations should fail or reopen
        # This is implementation-specific


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_vector_store_sqlite_vec(self):
        """Test getting SqliteVecStore via factory."""
        pytest.importorskip("sqlite_vec")
        from agent_runtime_core.vectorstore import get_vector_store

        store = get_vector_store("sqlite_vec", path=":memory:")
        assert store is not None

    def test_get_vector_store_unknown_backend(self):
        """Test error for unknown backend."""
        from agent_runtime_core.vectorstore import get_vector_store

        with pytest.raises(ValueError, match="Unknown vector store backend"):
            get_vector_store("unknown_backend")

    def test_get_embedding_client_openai(self):
        """Test getting OpenAI embedding client via factory."""
        pytest.importorskip("openai")
        from agent_runtime_core.vectorstore import get_embedding_client

        client = get_embedding_client("openai")
        assert client is not None
        assert client.model_name == "text-embedding-3-small"
        assert client.dimensions == 1536

    def test_get_embedding_client_unknown_provider(self):
        """Test error for unknown provider."""
        from agent_runtime_core.vectorstore import get_embedding_client

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_client("unknown_provider")


class TestEmbeddingClients:
    """Tests for embedding client implementations."""

    def test_openai_embeddings_dimensions(self):
        """Test OpenAI embedding dimensions."""
        pytest.importorskip("openai")
        from agent_runtime_core.vectorstore import OpenAIEmbeddings

        client = OpenAIEmbeddings(model="text-embedding-3-small")
        assert client.dimensions == 1536
        assert client.model_name == "text-embedding-3-small"

        client = OpenAIEmbeddings(model="text-embedding-3-large")
        assert client.dimensions == 3072

        # Custom dimensions
        client = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
        assert client.dimensions == 512

    def test_vertex_embeddings_dimensions(self):
        """Test Vertex AI embedding dimensions."""
        # Don't require google-cloud-aiplatform for this test
        from agent_runtime_core.vectorstore import VertexAIEmbeddings

        client = VertexAIEmbeddings(model="text-embedding-004")
        assert client.dimensions == 768
        assert client.model_name == "text-embedding-004"

