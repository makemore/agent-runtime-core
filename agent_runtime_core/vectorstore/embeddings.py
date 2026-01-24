"""
Embedding client interfaces and implementations.

Provides abstract interface for generating embeddings and concrete implementations
for OpenAI and Vertex AI embedding models.
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingClient(ABC):
    """
    Abstract interface for generating embeddings.

    Embedding clients convert text into vector representations that can be
    stored in a VectorStore for similarity search.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors in the same order as input
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    async def close(self) -> None:
        """Close any connections. Override if needed."""
        pass


class OpenAIEmbeddings(EmbeddingClient):
    """
    OpenAI embedding client using text-embedding-3-small or text-embedding-3-large.

    Requires: pip install openai
    """

    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedding client.

        Args:
            model: Model name (text-embedding-3-small, text-embedding-3-large)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            dimensions: Optional dimension override for text-embedding-3-* models
        """
        self._model = model
        self._api_key = api_key
        self._dimensions_override = dimensions
        self._client: Optional["openai.AsyncOpenAI"] = None  # type: ignore

    def _get_client(self) -> "openai.AsyncOpenAI":  # type: ignore
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        if self._dimensions_override:
            return self._dimensions_override
        return self.MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        client = self._get_client()
        kwargs = {"model": self._model, "input": text}
        if self._dimensions_override and self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions_override
        response = await client.embeddings.create(**kwargs)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        client = self._get_client()
        kwargs = {"model": self._model, "input": texts}
        if self._dimensions_override and self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions_override
        response = await client.embeddings.create(**kwargs)
        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class VertexAIEmbeddings(EmbeddingClient):
    """
    Vertex AI embedding client using text-embedding-004 or text-multilingual-embedding-002.

    Requires: pip install google-cloud-aiplatform
    """

    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        "text-multilingual-embedding-002": 768,
        "textembedding-gecko@003": 768,
        "textembedding-gecko-multilingual@001": 768,
    }

    def __init__(
        self,
        model: str = "text-embedding-004",
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        """
        Initialize Vertex AI embedding client.

        Args:
            model: Model name (text-embedding-004, text-multilingual-embedding-002)
            project_id: Google Cloud project ID (uses default if not provided)
            location: Google Cloud region
        """
        self._model = model
        self._project_id = project_id
        self._location = location
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK if not already done."""
        if self._initialized:
            return
        try:
            from google.cloud import aiplatform
        except ImportError:
            raise ImportError(
                "Google Cloud AI Platform package not installed. "
                "Install with: pip install google-cloud-aiplatform"
            )
        aiplatform.init(project=self._project_id, location=self._location)
        self._initialized = True

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.MODEL_DIMENSIONS.get(self._model, 768)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        self._ensure_initialized()
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self._model)
        # Run in executor since Vertex AI SDK is synchronous
        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: model.get_embeddings([text]))
        return embeddings[0].values

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        self._ensure_initialized()
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self._model)
        # Run in executor since Vertex AI SDK is synchronous
        import asyncio

        loop = asyncio.get_event_loop()
        # Vertex AI has a limit of 250 texts per batch
        batch_size = 250
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await loop.run_in_executor(
                None, lambda b=batch: model.get_embeddings(b)
            )
            all_embeddings.extend([e.values for e in embeddings])
        return all_embeddings

