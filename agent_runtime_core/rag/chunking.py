"""
Text chunking utilities for RAG.

Provides functions to split text into chunks suitable for embedding and retrieval.
This module has no external dependencies and can be used standalone.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    chunk_size: int = 500
    """Target size of each chunk in characters."""
    
    chunk_overlap: int = 50
    """Number of characters to overlap between chunks."""
    
    separator: str = "\n\n"
    """Primary separator to split on (paragraphs by default)."""
    
    fallback_separators: list[str] = None
    """Fallback separators if primary doesn't work: ["\n", ". ", " "]"""
    
    min_chunk_size: int = 100
    """Minimum chunk size - smaller chunks are merged with neighbors."""
    
    def __post_init__(self):
        if self.fallback_separators is None:
            self.fallback_separators = ["\n", ". ", " "]


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    
    text: str
    """The chunk text content."""
    
    index: int
    """Index of this chunk (0-based)."""
    
    start_char: int
    """Starting character position in original text."""
    
    end_char: int
    """Ending character position in original text."""
    
    metadata: dict = None
    """Optional metadata for this chunk."""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
    metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """
    Split text into chunks suitable for embedding.
    
    Uses a recursive approach:
    1. Try to split on primary separator (paragraphs)
    2. If chunks are too large, split on fallback separators
    3. Merge small chunks with neighbors
    4. Add overlap between chunks
    
    Args:
        text: The text to chunk
        config: Chunking configuration (uses defaults if not provided)
        metadata: Optional metadata to include in each chunk
        
    Returns:
        List of TextChunk objects
    """
    if config is None:
        config = ChunkingConfig()
    
    if metadata is None:
        metadata = {}
    
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = text.strip()
    
    # Split into initial segments using primary separator
    segments = _split_on_separator(text, config.separator)
    
    # Recursively split segments that are too large
    all_separators = [config.separator] + config.fallback_separators
    segments = _recursive_split(segments, all_separators, config.chunk_size)
    
    # Merge small segments
    segments = _merge_small_segments(segments, config.min_chunk_size, config.chunk_size)
    
    # Create chunks with overlap
    chunks = _create_chunks_with_overlap(
        segments, 
        config.chunk_overlap,
        text,
        metadata,
    )
    
    return chunks


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text on a separator, keeping non-empty segments."""
    if separator == ". ":
        # Special handling for sentence splitting - keep the period
        parts = re.split(r'(?<=\.)\s+', text)
    else:
        parts = text.split(separator)
    return [p.strip() for p in parts if p.strip()]


def _recursive_split(
    segments: list[str],
    separators: list[str],
    max_size: int,
    sep_index: int = 0,
) -> list[str]:
    """Recursively split segments that are too large."""
    if sep_index >= len(separators):
        # No more separators - just return as is (will be split by character if needed)
        return segments
    
    result = []
    separator = separators[sep_index]
    
    for segment in segments:
        if len(segment) <= max_size:
            result.append(segment)
        else:
            # Try to split on this separator
            sub_segments = _split_on_separator(segment, separator)
            if len(sub_segments) > 1:
                # Recursively process sub-segments
                result.extend(_recursive_split(sub_segments, separators, max_size, sep_index))
            else:
                # This separator didn't help, try next one
                result.extend(_recursive_split([segment], separators, max_size, sep_index + 1))
    
    return result


def _merge_small_segments(
    segments: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Merge segments that are too small with their neighbors."""
    if not segments:
        return []

    result = []
    current = segments[0]

    for segment in segments[1:]:
        combined = current + "\n\n" + segment
        if len(current) < min_size and len(combined) <= max_size:
            # Merge with current
            current = combined
        else:
            # Save current and start new
            result.append(current)
            current = segment

    result.append(current)
    return result


def _create_chunks_with_overlap(
    segments: list[str],
    overlap: int,
    original_text: str,
    base_metadata: dict,
) -> list[TextChunk]:
    """Create TextChunk objects with overlap between chunks."""
    chunks = []
    current_pos = 0

    for i, segment in enumerate(segments):
        # Find the actual position in original text
        start_pos = original_text.find(segment[:50], current_pos)
        if start_pos == -1:
            start_pos = current_pos

        end_pos = start_pos + len(segment)

        # Add overlap from previous chunk if not first
        if i > 0 and overlap > 0:
            # Get overlap text from end of previous segment
            prev_segment = segments[i - 1]
            overlap_text = prev_segment[-overlap:] if len(prev_segment) > overlap else prev_segment
            segment_with_overlap = overlap_text + " " + segment
        else:
            segment_with_overlap = segment

        chunk = TextChunk(
            text=segment_with_overlap,
            index=i,
            start_char=start_pos,
            end_char=end_pos,
            metadata={
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(segments),
            },
        )
        chunks.append(chunk)
        current_pos = end_pos

    return chunks

