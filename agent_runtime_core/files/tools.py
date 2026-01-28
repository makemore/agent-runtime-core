"""
File read/write tools for agents.

Provides sandboxed file access tools that agents can use to read and write files.
All file operations are restricted to configured allowed directories.

Example:
    from agent_runtime_core.files.tools import FileTools, FileToolsConfig

    config = FileToolsConfig(
        allowed_directories=["/app/uploads", "/app/outputs"],
        max_file_size_bytes=50 * 1024 * 1024,  # 50MB
    )
    tools = FileTools(config)

    # Read a file
    result = await tools.read_file("document.pdf")

    # Write a file
    await tools.write_file("output.txt", "Hello, world!")
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import base64

from .base import FileProcessorRegistry, ProcessingOptions, ProcessedFile, FileType


@dataclass
class FileToolsConfig:
    """Configuration for file tools."""
    # Sandboxing
    allowed_directories: list[str] = field(default_factory=lambda: ["."])

    # Size limits
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100MB default
    max_write_size_bytes: int = 100 * 1024 * 1024  # 100MB default

    # Processing options
    use_ocr: bool = False
    ocr_provider: Optional[str] = None
    use_vision: bool = False
    vision_provider: Optional[str] = None

    # Write options
    allow_overwrite: bool = False
    create_directories: bool = True


def get_file_read_schema() -> dict[str, Any]:
    """Get the tool schema for file_read in OpenAI format."""
    from ..tools import ToolSchemaBuilder

    return (
        ToolSchemaBuilder("file_read")
        .description(
            "Read and process a file. Extracts text content from various file types "
            "including PDF, DOCX, images (with optional OCR), spreadsheets, and text files. "
            "Returns the extracted text and metadata."
        )
        .param("path", "string", "Path to the file to read", required=True)
        .param("use_ocr", "boolean", "Use OCR for images/scanned documents", default=False)
        .param("ocr_provider", "string", "OCR provider to use",
               enum=["tesseract", "google", "aws", "azure"])
        .param("use_vision", "boolean", "Use AI vision for image analysis", default=False)
        .param("vision_provider", "string", "Vision AI provider to use",
               enum=["openai", "anthropic", "gemini"])
        .param("vision_prompt", "string", "Custom prompt for vision analysis")
        .to_openai_format()
    )


def get_file_write_schema() -> dict[str, Any]:
    """Get the tool schema for file_write in OpenAI format."""
    from ..tools import ToolSchemaBuilder

    return (
        ToolSchemaBuilder("file_write")
        .description(
            "Write content to a file. Can write text content or base64-encoded binary data. "
            "The file path must be within allowed directories."
        )
        .param("path", "string", "Path where the file should be written", required=True)
        .param("content", "string", "Content to write (text or base64 for binary)", required=True)
        .param("encoding", "string", "Content encoding", enum=["text", "base64"], default="text")
        .param("overwrite", "boolean", "Whether to overwrite existing files", default=False)
        .to_openai_format()
    )


class FileTools:
    """
    File read/write tools for agents with sandboxing.

    All file operations are restricted to configured allowed directories.
    """

    def __init__(
        self,
        config: Optional[FileToolsConfig] = None,
        registry: Optional[FileProcessorRegistry] = None,
    ):
        self.config = config or FileToolsConfig()
        self.registry = registry
        if self.registry is None:
            self.registry = FileProcessorRegistry()
            self.registry.auto_register()

    def _resolve_path(self, path: str) -> Path:
        """Resolve and validate a file path against allowed directories."""
        # Resolve to absolute path
        resolved = Path(path).resolve()

        # Check if path is within any allowed directory
        for allowed_dir in self.config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                resolved.relative_to(allowed_path)
                return resolved
            except ValueError:
                continue

        raise PermissionError(
            f"Access denied: '{path}' is not within allowed directories. "
            f"Allowed: {self.config.allowed_directories}"
        )

    async def read_file(
        self,
        path: str,
        use_ocr: bool = False,
        ocr_provider: Optional[str] = None,
        use_vision: bool = False,
        vision_provider: Optional[str] = None,
        vision_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Read and process a file.

        Args:
            path: Path to file
            use_ocr: Whether to use OCR for images/scanned documents
            ocr_provider: OCR provider (tesseract, google, aws, azure)
            use_vision: Whether to use AI vision analysis
            vision_provider: Vision provider (openai, anthropic, gemini)
            vision_prompt: Custom prompt for vision analysis

        Returns:
            Dict with extracted text, metadata, and processing info
        """
        resolved_path = self._resolve_path(path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check file size
        file_size = resolved_path.stat().st_size
        if file_size > self.config.max_file_size_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds limit "
                f"({self.config.max_file_size_bytes} bytes)"
            )

        # Read file content
        content = resolved_path.read_bytes()

        # Build processing options
        options = ProcessingOptions(
            max_size_bytes=self.config.max_file_size_bytes,
            use_ocr=use_ocr or self.config.use_ocr,
            ocr_provider=ocr_provider or self.config.ocr_provider,
            use_vision=use_vision or self.config.use_vision,
            vision_provider=vision_provider or self.config.vision_provider,
            vision_prompt=vision_prompt,
        )

        # Process file
        result = await self.registry.process(resolved_path.name, content, options)

        # Return as dict for tool response
        return {
            "filename": result.filename,
            "file_type": result.file_type.value,
            "mime_type": result.mime_type,
            "size_bytes": result.size_bytes,
            "text": result.text,
            "metadata": result.metadata,
            "ocr_text": result.ocr_text,
            "vision_description": result.vision_description,
            "warnings": result.warnings,
        }

    async def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "text",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: Path where the file should be written
            content: Content to write (text or base64 for binary)
            encoding: Content encoding ("text" or "base64")
            overwrite: Whether to overwrite existing files

        Returns:
            Dict with file info after writing
        """
        resolved_path = self._resolve_path(path)

        # Check if file exists and overwrite is allowed
        if resolved_path.exists():
            if not (overwrite or self.config.allow_overwrite):
                raise FileExistsError(
                    f"File already exists: {path}. Set overwrite=True to replace."
                )

        # Decode content
        if encoding == "base64":
            try:
                data = base64.b64decode(content)
            except Exception as e:
                raise ValueError(f"Invalid base64 content: {e}")
        else:
            data = content.encode("utf-8")

        # Check size limit
        if len(data) > self.config.max_write_size_bytes:
            raise ValueError(
                f"Content size ({len(data)} bytes) exceeds write limit "
                f"({self.config.max_write_size_bytes} bytes)"
            )

        # Create parent directories if needed
        if self.config.create_directories:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        resolved_path.write_bytes(data)

        return {
            "success": True,
            "path": str(resolved_path),
            "size_bytes": len(data),
            "encoding": encoding,
            "overwritten": resolved_path.exists() and (overwrite or self.config.allow_overwrite),
        }

    async def list_files(
        self,
        directory: str = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        List files in a directory.

        Args:
            directory: Directory to list (must be within allowed directories)
            pattern: Glob pattern to filter files
            recursive: Whether to search recursively

        Returns:
            Dict with list of files
        """
        resolved_dir = self._resolve_path(directory)

        if not resolved_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        if recursive:
            files = list(resolved_dir.rglob(pattern))
        else:
            files = list(resolved_dir.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]

        return {
            "directory": str(resolved_dir),
            "pattern": pattern,
            "recursive": recursive,
            "count": len(files),
            "files": [
                {
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
                for f in files[:100]  # Limit to 100 files
            ],
            "truncated": len(files) > 100,
        }


def get_file_list_schema() -> dict[str, Any]:
    """Get the tool schema for file_list in OpenAI format."""
    from ..tools import ToolSchemaBuilder

    return (
        ToolSchemaBuilder("file_list")
        .description(
            "List files in a directory. Returns file names, sizes, and modification times. "
            "The directory must be within allowed directories."
        )
        .param("directory", "string", "Directory to list", default=".")
        .param("pattern", "string", "Glob pattern to filter files", default="*")
        .param("recursive", "boolean", "Search recursively", default=False)
        .to_openai_format()
    )
