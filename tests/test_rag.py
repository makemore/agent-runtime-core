"""
Tests for RAG (Retrieval Augmented Generation) module.

These tests verify the portable RAG implementation that works without Django.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestChunking:
    """Tests for text chunking utilities."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        from agent_runtime_core.rag import chunk_text, ChunkingConfig
        
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text)
        
        assert len(chunks) > 0
        # All chunks should have text
        for chunk in chunks:
            assert chunk.text
            assert chunk.index >= 0

    def test_chunk_text_with_config(self):
        """Test chunking with custom configuration."""
        from agent_runtime_core.rag import chunk_text, ChunkingConfig

        # Create text with natural split points (paragraphs)
        text = "\n\n".join(["This is paragraph number " + str(i) + "." for i in range(20)])
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
        chunks = chunk_text(text, config=config)

        # Should create multiple chunks
        assert len(chunks) > 1

    def test_chunk_text_with_metadata(self):
        """Test chunking preserves metadata."""
        from agent_runtime_core.rag import chunk_text
        
        text = "Some text content."
        metadata = {"source": "test", "doc_id": "123"}
        chunks = chunk_text(text, metadata=metadata)
        
        assert len(chunks) > 0
        assert chunks[0].metadata.get("source") == "test"
        assert chunks[0].metadata.get("doc_id") == "123"

    def test_chunk_text_empty(self):
        """Test chunking empty text returns empty list."""
        from agent_runtime_core.rag import chunk_text
        
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == [] if chunk_text(None) is not None else True

    def test_text_chunk_dataclass(self):
        """Test TextChunk dataclass."""
        from agent_runtime_core.rag import TextChunk
        
        chunk = TextChunk(
            text="Hello world",
            index=0,
            start_char=0,
            end_char=11,
            metadata={"key": "value"},
        )
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.metadata == {"key": "value"}


class TestKnowledgeIndexer:
    """Tests for KnowledgeIndexer."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = AsyncMock()
        store.add_batch = AsyncMock()
        store.delete_by_filter = AsyncMock(return_value=0)
        store.close = AsyncMock()
        return store

    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock embedding client."""
        client = AsyncMock()
        # Return fake embeddings
        client.embed = AsyncMock(return_value=[0.1] * 1536)
        client.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def indexer(self, mock_vector_store, mock_embedding_client):
        """Create a KnowledgeIndexer with mocked dependencies."""
        from agent_runtime_core.rag import KnowledgeIndexer
        return KnowledgeIndexer(
            vector_store=mock_vector_store,
            embedding_client=mock_embedding_client,
        )

    @pytest.mark.asyncio
    async def test_index_text_success(self, indexer, mock_vector_store, mock_embedding_client):
        """Test successful text indexing."""
        result = await indexer.index_text(
            text="This is some test content for indexing.",
            source_id="doc-1",
            metadata={"name": "Test Document"},
        )
        
        assert result.status == "success"
        assert result.source_id == "doc-1"
        assert result.chunks_indexed > 0
        
        # Verify embedding was called
        mock_embedding_client.embed_batch.assert_called_once()
        
        # Verify vectors were stored
        mock_vector_store.add_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_text_empty_content(self, indexer):
        """Test indexing empty content returns error."""
        result = await indexer.index_text(
            text="",
            source_id="doc-1",
        )
        
        assert result.status == "error"
        assert "No content" in result.error

    @pytest.mark.asyncio
    async def test_delete_source(self, indexer, mock_vector_store):
        """Test deleting vectors for a source."""
        mock_vector_store.delete_by_filter.return_value = 5
        
        deleted = await indexer.delete_source("doc-1")
        
        assert deleted == 5
        mock_vector_store.delete_by_filter.assert_called_once_with({"source_id": "doc-1"})


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        from agent_runtime_core.vectorstore import VectorSearchResult

        store = AsyncMock()
        store.search = AsyncMock(return_value=[
            VectorSearchResult(
                id="chunk-1",
                content="Relevant content here",
                score=0.95,
                metadata={"source_id": "doc-1", "name": "Test Doc"},
            ),
        ])
        store.close = AsyncMock()
        return store

    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock embedding client."""
        client = AsyncMock()
        client.embed = AsyncMock(return_value=[0.1] * 1536)
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_client):
        """Create a KnowledgeRetriever with mocked dependencies."""
        from agent_runtime_core.rag import KnowledgeRetriever
        return KnowledgeRetriever(
            vector_store=mock_vector_store,
            embedding_client=mock_embedding_client,
        )

    @pytest.mark.asyncio
    async def test_retrieve_success(self, retriever, mock_vector_store, mock_embedding_client):
        """Test successful retrieval."""
        results = await retriever.retrieve("What is the weather?", top_k=5)

        assert len(results) == 1
        assert results[0].content == "Relevant content here"
        assert results[0].score == 0.95

        # Verify embedding was called
        mock_embedding_client.embed.assert_called_once_with("What is the weather?")

        # Verify search was called
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self, retriever, mock_vector_store, mock_embedding_client):
        """Test retrieval with metadata filter."""
        await retriever.retrieve(
            "Search query",
            top_k=3,
            filter={"source_id": "doc-1"},
        )

        # Verify filter was passed to search
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs.get("filter") == {"source_id": "doc-1"}

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, retriever, mock_vector_store):
        """Test retrieval with no results."""
        mock_vector_store.search.return_value = []

        results = await retriever.retrieve("Unknown query")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self, retriever, mock_vector_store):
        """Test retrieval filters by similarity threshold."""
        from agent_runtime_core.vectorstore import VectorSearchResult

        mock_vector_store.search.return_value = [
            VectorSearchResult(id="1", content="High score", score=0.9, metadata={}),
            VectorSearchResult(id="2", content="Low score", score=0.3, metadata={}),
        ]

        results = await retriever.retrieve("Query", similarity_threshold=0.5)

        # Only high score result should be returned
        assert len(results) == 1
        assert results[0].content == "High score"
