"""
Base classes for file processing.

Provides the FileProcessor abstract base class and registry pattern
for pluggable file type handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type, Union
import mimetypes


class FileType(str, Enum):
    """Supported file types."""
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    UNKNOWN = "unknown"


@dataclass
class ProcessingOptions:
    """Options for file processing."""
    # General options
    max_size_bytes: int = 100 * 1024 * 1024  # 100MB default
    extract_text: bool = True
    extract_metadata: bool = True
    
    # OCR options
    use_ocr: bool = False
    ocr_provider: Optional[str] = None  # tesseract, google, aws, azure
    ocr_language: str = "eng"
    
    # Vision AI options
    use_vision: bool = False
    vision_provider: Optional[str] = None  # openai, anthropic, gemini
    vision_prompt: Optional[str] = None  # Custom prompt for vision analysis
    
    # Image options
    generate_thumbnail: bool = True
    thumbnail_size: tuple[int, int] = (200, 200)
    
    # PDF options
    pdf_extract_images: bool = False
    pdf_page_limit: Optional[int] = None  # Limit pages to process
    
    # Additional provider-specific options
    extra: dict = field(default_factory=dict)


@dataclass
class ProcessedFile:
    """Result of processing a file."""
    # Core data
    filename: str
    file_type: FileType
    mime_type: str
    size_bytes: int
    
    # Extracted content
    text: str = ""
    text_chunks: list[str] = field(default_factory=list)  # For chunked processing
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    # Visual data
    thumbnail_base64: Optional[str] = None
    preview_url: Optional[str] = None
    
    # OCR/Vision results
    ocr_text: Optional[str] = None
    vision_description: Optional[str] = None
    vision_analysis: Optional[dict] = None
    
    # Processing info
    processor_used: str = ""
    processing_time_ms: float = 0
    warnings: list[str] = field(default_factory=list)
    
    # Raw data (optional, for further processing)
    raw_content: Optional[bytes] = None


class FileProcessor(ABC):
    """
    Abstract base class for file processors.
    
    Subclass this to create processors for specific file types.
    Each processor declares which file types and MIME types it handles.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this processor."""
        ...
    
    @property
    @abstractmethod
    def supported_types(self) -> list[FileType]:
        """List of FileType enums this processor handles."""
        ...
    
    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of file extensions this processor handles (e.g., ['.pdf', '.PDF'])."""
        ...
    
    @property
    def supported_mime_types(self) -> list[str]:
        """List of MIME types this processor handles. Override if needed."""
        return []
    
    @abstractmethod
    async def process(
        self,
        content: bytes,
        filename: str,
        options: ProcessingOptions,
    ) -> ProcessedFile:
        """
        Process file content and extract text/metadata.
        
        Args:
            content: Raw file bytes
            filename: Original filename
            options: Processing options
            
        Returns:
            ProcessedFile with extracted content
        """
        ...
    
    def can_process(self, filename: str, mime_type: Optional[str] = None) -> bool:
        """Check if this processor can handle the given file."""
        ext = Path(filename).suffix.lower()
        if ext in [e.lower() for e in self.supported_extensions]:
            return True
        if mime_type and mime_type in self.supported_mime_types:
            return True
        return False


class FileProcessorRegistry:
    """
    Registry of file processors.
    
    Manages processor registration and selection based on file type.
    """
    
    def __init__(self):
        self._processors: dict[str, FileProcessor] = {}
        self._type_map: dict[FileType, list[str]] = {}
        self._extension_map: dict[str, str] = {}
    
    def register(self, processor: FileProcessor) -> None:
        """Register a file processor."""
        self._processors[processor.name] = processor
        
        # Map file types to processor
        for file_type in processor.supported_types:
            if file_type not in self._type_map:
                self._type_map[file_type] = []
            self._type_map[file_type].append(processor.name)
        
        # Map extensions to processor
        for ext in processor.supported_extensions:
            self._extension_map[ext.lower()] = processor.name

    def get(self, name: str) -> Optional[FileProcessor]:
        """Get a processor by name."""
        return self._processors.get(name)

    def get_for_file(
        self,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> Optional[FileProcessor]:
        """Get the best processor for a file."""
        ext = Path(filename).suffix.lower()

        # Try extension first
        if ext in self._extension_map:
            return self._processors[self._extension_map[ext]]

        # Try MIME type
        if mime_type:
            for processor in self._processors.values():
                if mime_type in processor.supported_mime_types:
                    return processor

        # Guess MIME type from filename
        guessed_mime, _ = mimetypes.guess_type(filename)
        if guessed_mime:
            for processor in self._processors.values():
                if guessed_mime in processor.supported_mime_types:
                    return processor

        return None

    async def process(
        self,
        filename: str,
        content: bytes,
        options: Optional[ProcessingOptions] = None,
        mime_type: Optional[str] = None,
    ) -> ProcessedFile:
        """
        Process a file using the appropriate processor.

        Args:
            filename: Original filename
            content: Raw file bytes
            options: Processing options (uses defaults if not provided)
            mime_type: Optional MIME type hint

        Returns:
            ProcessedFile with extracted content

        Raises:
            ValueError: If no processor found for file type
            ValueError: If file exceeds size limit
        """
        if options is None:
            options = ProcessingOptions()

        # Check size limit
        if len(content) > options.max_size_bytes:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds limit "
                f"({options.max_size_bytes} bytes)"
            )

        # Find processor
        processor = self.get_for_file(filename, mime_type)
        if not processor:
            raise ValueError(f"No processor found for file: {filename}")

        # Process
        return await processor.process(content, filename, options)

    def list_processors(self) -> list[FileProcessor]:
        """List all registered processors."""
        return list(self._processors.values())

    def supported_extensions(self) -> list[str]:
        """List all supported file extensions."""
        return list(self._extension_map.keys())

    def auto_register(self) -> None:
        """
        Auto-register all available processors.

        Registers built-in processors and checks for optional dependencies.
        """
        from .processors import (
            TextFileProcessor,
            PDFProcessor,
            ImageProcessor,
            DocxProcessor,
            XlsxProcessor,
            CsvProcessor,
        )

        # Always available
        self.register(TextFileProcessor())
        self.register(CsvProcessor())

        # Check for optional dependencies
        try:
            import pypdf
            self.register(PDFProcessor())
        except ImportError:
            pass

        try:
            from PIL import Image
            self.register(ImageProcessor())
        except ImportError:
            pass

        try:
            import docx
            self.register(DocxProcessor())
        except ImportError:
            pass

        try:
            import openpyxl
            self.register(XlsxProcessor())
        except ImportError:
            pass


def detect_file_type(filename: str, content: Optional[bytes] = None) -> FileType:
    """
    Detect file type from filename and optionally content.

    Args:
        filename: Filename with extension
        content: Optional file content for magic number detection

    Returns:
        Detected FileType
    """
    ext = Path(filename).suffix.lower()

    extension_map = {
        ".txt": FileType.TEXT,
        ".text": FileType.TEXT,
        ".log": FileType.TEXT,
        ".pdf": FileType.PDF,
        ".png": FileType.IMAGE,
        ".jpg": FileType.IMAGE,
        ".jpeg": FileType.IMAGE,
        ".gif": FileType.IMAGE,
        ".webp": FileType.IMAGE,
        ".bmp": FileType.IMAGE,
        ".docx": FileType.DOCX,
        ".doc": FileType.DOCX,
        ".xlsx": FileType.XLSX,
        ".xls": FileType.XLSX,
        ".csv": FileType.CSV,
        ".json": FileType.JSON,
        ".md": FileType.MARKDOWN,
        ".markdown": FileType.MARKDOWN,
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
    }

    return extension_map.get(ext, FileType.UNKNOWN)

