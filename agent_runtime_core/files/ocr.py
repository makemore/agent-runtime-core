"""
OCR (Optical Character Recognition) providers.

Supports multiple OCR backends:
- Tesseract (local, free)
- Google Cloud Vision
- AWS Textract
- Azure Document Intelligence

All providers are optional and checked at runtime.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import base64


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    blocks: list[dict] = field(default_factory=list)  # Structured text blocks
    raw_response: Optional[Any] = None


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
    
    @abstractmethod
    async def extract_text(
        self,
        image_bytes: bytes,
        language: str = "eng",
        **kwargs,
    ) -> OCRResult:
        """
        Extract text from an image.
        
        Args:
            image_bytes: Raw image bytes
            language: Language hint (ISO 639-3 code for Tesseract, varies by provider)
            **kwargs: Provider-specific options
            
        Returns:
            OCRResult with extracted text
        """
        ...
    
    def is_available(self) -> bool:
        """Check if this provider is available (dependencies installed, configured)."""
        return True


class TesseractOCR(OCRProvider):
    """
    Tesseract OCR provider (local, free).
    
    Requires: pytesseract, tesseract-ocr system package
    Install: pip install pytesseract
             brew install tesseract (macOS) or apt install tesseract-ocr (Linux)
    """
    
    @property
    def name(self) -> str:
        return "tesseract"
    
    def is_available(self) -> bool:
        try:
            import pytesseract
            # Try to get version to verify tesseract is installed
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    async def extract_text(
        self,
        image_bytes: bytes,
        language: str = "eng",
        **kwargs,
    ) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pytesseract and Pillow are required for Tesseract OCR. "
                "Install with: pip install pytesseract Pillow"
            )
        
        import io
        
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get detailed data
        data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)
        
        # Extract text
        text = pytesseract.image_to_string(img, lang=language)
        
        # Calculate average confidence
        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Build blocks
        blocks = []
        current_block = {"text": "", "words": []}
        for i, word in enumerate(data["text"]):
            if word.strip():
                current_block["words"].append({
                    "text": word,
                    "confidence": data["conf"][i],
                    "bbox": {
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    },
                })
        
        return OCRResult(
            text=text.strip(),
            confidence=avg_confidence,
            language=language,
            blocks=blocks,
        )


class GoogleVisionOCR(OCRProvider):
    """
    Google Cloud Vision OCR provider.
    
    Requires: google-cloud-vision
    Install: pip install google-cloud-vision
    Auth: Set GOOGLE_APPLICATION_CREDENTIALS environment variable
    """
    
    @property
    def name(self) -> str:
        return "google_vision"
    
    def is_available(self) -> bool:
        try:
            from google.cloud import vision
            return True
        except ImportError:
            return False
    
    async def extract_text(
        self,
        image_bytes: bytes,
        language: str = "en",
        **kwargs,
    ) -> OCRResult:
        try:
            from google.cloud import vision
        except ImportError:
            raise ImportError(
                "google-cloud-vision is required for Google Vision OCR. "
                "Install with: pip install google-cloud-vision"
            )

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)

        # Request text detection
        response = client.text_detection(
            image=image,
            image_context=vision.ImageContext(language_hints=[language]),
        )

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        # Extract full text
        texts = response.text_annotations
        full_text = texts[0].description if texts else ""

        # Build blocks from individual text annotations
        blocks = []
        for text in texts[1:]:  # Skip first (full text)
            vertices = text.bounding_poly.vertices
            blocks.append({
                "text": text.description,
                "bbox": {
                    "left": vertices[0].x if vertices else 0,
                    "top": vertices[0].y if vertices else 0,
                    "right": vertices[2].x if len(vertices) > 2 else 0,
                    "bottom": vertices[2].y if len(vertices) > 2 else 0,
                },
            })

        return OCRResult(
            text=full_text.strip(),
            confidence=None,  # Google Vision doesn't provide overall confidence
            language=language,
            blocks=blocks,
            raw_response=response,
        )


class AWSTextractOCR(OCRProvider):
    """
    AWS Textract OCR provider.

    Requires: boto3
    Install: pip install boto3
    Auth: Configure AWS credentials (environment variables, ~/.aws/credentials, or IAM role)
    """

    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name

    @property
    def name(self) -> str:
        return "aws_textract"

    def is_available(self) -> bool:
        try:
            import boto3
            return True
        except ImportError:
            return False

    async def extract_text(
        self,
        image_bytes: bytes,
        language: str = "en",
        **kwargs,
    ) -> OCRResult:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Textract OCR. "
                "Install with: pip install boto3"
            )

        client = boto3.client("textract", region_name=self.region_name)

        # Call Textract
        response = client.detect_document_text(Document={"Bytes": image_bytes})

        # Extract text and blocks
        lines = []
        blocks = []
        confidences = []

        for block in response.get("Blocks", []):
            if block["BlockType"] == "LINE":
                lines.append(block.get("Text", ""))
                confidences.append(block.get("Confidence", 0))

                bbox = block.get("Geometry", {}).get("BoundingBox", {})
                blocks.append({
                    "text": block.get("Text", ""),
                    "confidence": block.get("Confidence"),
                    "bbox": {
                        "left": bbox.get("Left", 0),
                        "top": bbox.get("Top", 0),
                        "width": bbox.get("Width", 0),
                        "height": bbox.get("Height", 0),
                    },
                })

        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return OCRResult(
            text="\n".join(lines),
            confidence=avg_confidence,
            language=language,
            blocks=blocks,
            raw_response=response,
        )


class AzureDocumentOCR(OCRProvider):
    """
    Azure Document Intelligence (Form Recognizer) OCR provider.

    Requires: azure-ai-formrecognizer
    Install: pip install azure-ai-formrecognizer
    Auth: Set AZURE_FORM_RECOGNIZER_ENDPOINT and AZURE_FORM_RECOGNIZER_KEY environment variables
    """

    def __init__(self, endpoint: str | None = None, key: str | None = None):
        import os
        self.endpoint = endpoint or os.environ.get("AZURE_FORM_RECOGNIZER_ENDPOINT")
        self.key = key or os.environ.get("AZURE_FORM_RECOGNIZER_KEY")

    @property
    def name(self) -> str:
        return "azure_document"

    def is_available(self) -> bool:
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            return bool(self.endpoint and self.key)
        except ImportError:
            return False

    async def extract_text(
        self,
        image_bytes: bytes,
        language: str = "en",
        **kwargs,
    ) -> OCRResult:
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-ai-formrecognizer is required for Azure Document OCR. "
                "Install with: pip install azure-ai-formrecognizer"
            )

        if not self.endpoint or not self.key:
            raise ValueError(
                "Azure endpoint and key are required. "
                "Set AZURE_FORM_RECOGNIZER_ENDPOINT and AZURE_FORM_RECOGNIZER_KEY environment variables."
            )

        client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key),
        )

        # Analyze document
        poller = client.begin_analyze_document("prebuilt-read", image_bytes)
        result = poller.result()

        # Extract text
        lines = []
        blocks = []
        confidences = []

        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
                if line.spans:
                    confidences.append(getattr(line, "confidence", 0.9))

                # Get bounding box
                if line.polygon:
                    blocks.append({
                        "text": line.content,
                        "polygon": [(p.x, p.y) for p in line.polygon],
                    })

        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return OCRResult(
            text="\n".join(lines),
            confidence=avg_confidence,
            language=language,
            blocks=blocks,
            raw_response=result,
        )


# Registry of available OCR providers
OCR_PROVIDERS: dict[str, type[OCRProvider]] = {
    "tesseract": TesseractOCR,
    "google_vision": GoogleVisionOCR,
    "aws_textract": AWSTextractOCR,
    "azure_document": AzureDocumentOCR,
}


def get_ocr_provider(name: str, **kwargs) -> OCRProvider:
    """
    Get an OCR provider by name.

    Args:
        name: Provider name (tesseract, google_vision, aws_textract, azure_document)
        **kwargs: Provider-specific configuration

    Returns:
        Configured OCRProvider instance
    """
    if name not in OCR_PROVIDERS:
        raise ValueError(f"Unknown OCR provider: {name}. Available: {list(OCR_PROVIDERS.keys())}")
    return OCR_PROVIDERS[name](**kwargs)


def get_available_ocr_providers() -> list[str]:
    """Get list of available (installed and configured) OCR providers."""
    available = []
    for name, provider_class in OCR_PROVIDERS.items():
        try:
            provider = provider_class()
            if provider.is_available():
                available.append(name)
        except Exception:
            pass
    return available
