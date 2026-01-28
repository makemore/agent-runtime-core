"""
AI Vision providers for image analysis.

Supports multiple AI vision providers:
- OpenAI GPT-4 Vision
- Anthropic Claude Vision
- Google Gemini Vision

All providers are optional - install the corresponding library to use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import base64
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from AI vision analysis."""

    description: str
    """Natural language description of the image."""

    labels: list[str] = field(default_factory=list)
    """Detected labels/objects in the image."""

    raw_response: Any = None
    """Raw response from the vision provider."""

    model: str = ""
    """Model used for analysis."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage information."""


class VisionProvider(ABC):
    """Abstract base class for AI vision providers."""

    name: str = "base"

    @abstractmethod
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail.",
        mime_type: str = "image/png",
        **kwargs,
    ) -> VisionResult:
        """
        Analyze an image using AI vision.

        Args:
            image_data: Raw image bytes
            prompt: Question or instruction for the vision model
            mime_type: MIME type of the image
            **kwargs: Provider-specific options

        Returns:
            VisionResult with description and metadata
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this provider's dependencies are installed."""
        return False

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64."""
        return base64.b64encode(image_data).decode("utf-8")


class OpenAIVision(VisionProvider):
    """OpenAI GPT-4 Vision provider."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI library not installed. Install with: pip install openai"
                )
        return self._client

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail.",
        mime_type: str = "image/png",
        **kwargs,
    ) -> VisionResult:
        client = self._get_client()

        base64_image = self._encode_image(image_data)
        data_url = f"data:{mime_type};base64,{base64_image}"

        response = await client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )

        description = response.choices[0].message.content or ""

        return VisionResult(
            description=description,
            model=response.model,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )


class AnthropicVision(VisionProvider):
    """Anthropic Claude Vision provider."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None



    @classmethod
    def is_available(cls) -> bool:
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                )
        return self._client

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail.",
        mime_type: str = "image/png",
        **kwargs,
    ) -> VisionResult:
        client = self._get_client()

        base64_image = self._encode_image(image_data)

        response = await client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        description = response.content[0].text if response.content else ""

        return VisionResult(
            description=description,
            model=response.model,
            raw_response=response.model_dump(),
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )


class GeminiVision(VisionProvider):
    """Google Gemini Vision provider."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key
        self.model = model
        self._client = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "Google Generative AI library not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str = "Describe this image in detail.",
        mime_type: str = "image/png",
        **kwargs,
    ) -> VisionResult:
        client = self._get_client()

        # Gemini uses PIL Image or inline data
        image_part = {
            "mime_type": mime_type,
            "data": image_data,
        }

        # Gemini's generate_content is sync, wrap it
        import asyncio
        response = await asyncio.to_thread(
            client.generate_content,
            [prompt, image_part],
        )

        description = response.text if response.text else ""

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            }

        return VisionResult(
            description=description,
            model=kwargs.get("model", self.model),
            raw_response=response,
            usage=usage,
        )


# Registry of all vision providers
VISION_PROVIDERS: dict[str, type[VisionProvider]] = {
    "openai": OpenAIVision,
    "anthropic": AnthropicVision,
    "gemini": GeminiVision,
}


def get_vision_provider(
    name: str,
    **kwargs,
) -> VisionProvider:
    """
    Get a vision provider by name.

    Args:
        name: Provider name ('openai', 'anthropic', 'gemini')
        **kwargs: Provider-specific configuration

    Returns:
        Configured VisionProvider instance

    Raises:
        ValueError: If provider name is unknown
        ImportError: If provider dependencies are not installed
    """
    if name not in VISION_PROVIDERS:
        available = list(VISION_PROVIDERS.keys())
        raise ValueError(f"Unknown vision provider: {name}. Available: {available}")

    provider_class = VISION_PROVIDERS[name]

    if not provider_class.is_available():
        raise ImportError(
            f"Vision provider '{name}' dependencies not installed. "
            f"Check the provider documentation for installation instructions."
        )

    return provider_class(**kwargs)


def get_available_vision_providers() -> list[str]:
    """
    Get list of vision providers that have their dependencies installed.

    Returns:
        List of available provider names
    """
    return [
        name for name, cls in VISION_PROVIDERS.items()
        if cls.is_available()
    ]
