"""
File processing module for agent_runtime_core.

Provides pluggable file processors for reading various file types,
OCR integration, and AI vision capabilities.

Example:
    from agent_runtime_core.files import FileProcessorRegistry, process_file
    
    # Register processors
    registry = FileProcessorRegistry()
    registry.auto_register()  # Register all available processors
    
    # Process a file
    result = await registry.process("document.pdf", file_bytes)
    print(result.text)  # Extracted text
    print(result.metadata)  # File metadata
"""

from .base import (
    FileProcessor,
    FileProcessorRegistry,
    ProcessedFile,
    FileType,
    ProcessingOptions,
)
from .processors import (
    TextFileProcessor,
    PDFProcessor,
    ImageProcessor,
    DocxProcessor,
    XlsxProcessor,
    CsvProcessor,
)
from .ocr import (
    OCRProvider,
    TesseractOCR,
    GoogleVisionOCR,
    AWSTextractOCR,
    AzureDocumentOCR,
)
from .vision import (
    VisionProvider,
    OpenAIVision,
    AnthropicVision,
    GeminiVision,
)
from .tools import (
    FileTools,
    FileToolsConfig,
    get_file_read_schema,
    get_file_write_schema,
    get_file_list_schema,
)

__all__ = [
    # Base classes
    "FileProcessor",
    "FileProcessorRegistry",
    "ProcessedFile",
    "FileType",
    "ProcessingOptions",
    # Processors
    "TextFileProcessor",
    "PDFProcessor",
    "ImageProcessor",
    "DocxProcessor",
    "XlsxProcessor",
    "CsvProcessor",
    # OCR
    "OCRProvider",
    "TesseractOCR",
    "GoogleVisionOCR",
    "AWSTextractOCR",
    "AzureDocumentOCR",
    # Vision
    "VisionProvider",
    "OpenAIVision",
    "AnthropicVision",
    "GeminiVision",
    # Tools
    "FileTools",
    "FileToolsConfig",
    "get_file_read_schema",
    "get_file_write_schema",
    "get_file_list_schema",
]

