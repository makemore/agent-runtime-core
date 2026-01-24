"""
SQLite-vec vector store implementation.

Uses the sqlite-vec extension for vector similarity search.
Ideal for local development and small to medium datasets.

Requires: pip install sqlite-vec
"""

import json
import sqlite3
import struct
from typing import Optional

from agent_runtime_core.vectorstore.base import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
)


def _serialize_vector(vector: list[float]) -> bytes:
    """Serialize a vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_vector(data: bytes) -> list[float]:
    """Deserialize bytes to a vector."""
    n = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{n}f", data))


class SqliteVecStore(VectorStore):
    """
    Vector store using sqlite-vec extension.

    This implementation stores vectors in a SQLite database with the sqlite-vec
    extension for efficient similarity search. It's ideal for:
    - Local development
    - Small to medium datasets (up to millions of vectors)
    - Embedded applications without external dependencies

    The store creates two tables:
    - {table_name}_vec: Virtual table for vector storage and search
    - {table_name}_meta: Regular table for content and metadata
    """

    def __init__(
        self,
        path: str = ":memory:",
        table_name: str = "vectors",
    ):
        """
        Initialize SQLite-vec store.

        Args:
            path: Database path (":memory:" for in-memory, or file path)
            table_name: Base name for the tables
        """
        self._path = path
        self._table_name = table_name
        self._conn: Optional[sqlite3.Connection] = None
        self._dimensions: Optional[int] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            try:
                import sqlite_vec
            except ImportError:
                raise ImportError(
                    "sqlite-vec package not installed. Install with: pip install sqlite-vec"
                )
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        return self._conn

    def _ensure_tables(self, dimensions: int) -> None:
        """Ensure the required tables exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if tables already exist
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self._table_name}_meta'"
        )
        if cursor.fetchone() is not None:
            # Tables exist, verify dimensions match
            if self._dimensions is None:
                # Get dimensions from existing virtual table
                cursor.execute(f"PRAGMA table_info({self._table_name}_vec)")
                # The virtual table structure varies, so we'll trust the stored dimensions
                self._dimensions = dimensions
            return

        self._dimensions = dimensions

        # Create metadata table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name}_meta (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{{}}'
            )
        """)

        # Create virtual table for vectors
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table_name}_vec 
            USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dimensions}]
            )
        """)

        conn.commit()

    async def add(
        self,
        id: str,
        vector: list[float],
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector with its content and metadata."""
        self._ensure_tables(len(vector))
        conn = self._get_connection()
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata or {})
        vector_bytes = _serialize_vector(vector)

        # Insert or replace in both tables
        cursor.execute(
            f"INSERT OR REPLACE INTO {self._table_name}_meta (id, content, metadata) VALUES (?, ?, ?)",
            (id, content, metadata_json),
        )
        cursor.execute(
            f"INSERT OR REPLACE INTO {self._table_name}_vec (id, embedding) VALUES (?, ?)",
            (id, vector_bytes),
        )
        conn.commit()

    async def add_batch(
        self,
        items: list[tuple[str, list[float], str, Optional[dict]]],
    ) -> None:
        """Add multiple vectors efficiently."""
        if not items:
            return

        # Get dimensions from first item
        self._ensure_tables(len(items[0][1]))
        conn = self._get_connection()
        cursor = conn.cursor()

        meta_data = []
        vec_data = []
        for id, vector, content, metadata in items:
            meta_data.append((id, content, json.dumps(metadata or {})))
            vec_data.append((id, _serialize_vector(vector)))

        cursor.executemany(
            f"INSERT OR REPLACE INTO {self._table_name}_meta (id, content, metadata) VALUES (?, ?, ?)",
            meta_data,
        )
        cursor.executemany(
            f"INSERT OR REPLACE INTO {self._table_name}_vec (id, embedding) VALUES (?, ?)",
            vec_data,
        )
        conn.commit()

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        if self._dimensions is None:
            # No vectors added yet
            return []

        conn = self._get_connection()
        cursor = conn.cursor()
        query_bytes = _serialize_vector(query_vector)

        # sqlite-vec uses distance (lower = more similar), we convert to similarity score
        if filter:
            # Build filter conditions for metadata
            filter_conditions = []
            filter_values = []
            for key, value in filter.items():
                filter_conditions.append(f"json_extract(m.metadata, '$.{key}') = ?")
                filter_values.append(json.dumps(value) if not isinstance(value, str) else value)

            filter_sql = " AND ".join(filter_conditions)
            cursor.execute(
                f"""
                SELECT v.id, v.distance, m.content, m.metadata
                FROM {self._table_name}_vec v
                JOIN {self._table_name}_meta m ON v.id = m.id
                WHERE v.embedding MATCH ? AND k = ?
                AND {filter_sql}
                ORDER BY v.distance
                """,
                [query_bytes, limit] + filter_values,
            )
        else:
            cursor.execute(
                f"""
                SELECT v.id, v.distance, m.content, m.metadata
                FROM {self._table_name}_vec v
                JOIN {self._table_name}_meta m ON v.id = m.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (query_bytes, limit),
            )

        results = []
        for row in cursor.fetchall():
            id, distance, content, metadata_json = row
            # Convert distance to similarity score (1 / (1 + distance))
            score = 1.0 / (1.0 + distance)
            results.append(
                VectorSearchResult(
                    id=id,
                    content=content,
                    score=score,
                    metadata=json.loads(metadata_json),
                )
            )
        return results

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {self._table_name}_meta WHERE id = ?", (id,))
        deleted_meta = cursor.rowcount > 0

        cursor.execute(f"DELETE FROM {self._table_name}_vec WHERE id = ?", (id,))
        conn.commit()

        return deleted_meta

    async def delete_by_filter(self, filter: dict) -> int:
        """Delete vectors matching filter."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build filter conditions
        filter_conditions = []
        filter_values = []
        for key, value in filter.items():
            filter_conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            filter_values.append(json.dumps(value) if not isinstance(value, str) else value)

        filter_sql = " AND ".join(filter_conditions)

        # Get IDs to delete
        cursor.execute(
            f"SELECT id FROM {self._table_name}_meta WHERE {filter_sql}",
            filter_values,
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]

        if not ids_to_delete:
            return 0

        # Delete from both tables
        placeholders = ",".join("?" * len(ids_to_delete))
        cursor.execute(
            f"DELETE FROM {self._table_name}_meta WHERE id IN ({placeholders})",
            ids_to_delete,
        )
        cursor.execute(
            f"DELETE FROM {self._table_name}_vec WHERE id IN ({placeholders})",
            ids_to_delete,
        )
        conn.commit()

        return len(ids_to_delete)

    async def get(self, id: str) -> Optional[VectorRecord]:
        """Get a vector by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get metadata
        cursor.execute(
            f"SELECT content, metadata FROM {self._table_name}_meta WHERE id = ?",
            (id,),
        )
        meta_row = cursor.fetchone()
        if meta_row is None:
            return None

        content, metadata_json = meta_row

        # Get vector
        cursor.execute(
            f"SELECT embedding FROM {self._table_name}_vec WHERE id = ?",
            (id,),
        )
        vec_row = cursor.fetchone()
        if vec_row is None:
            return None

        vector = _deserialize_vector(vec_row[0])

        return VectorRecord(
            id=id,
            vector=vector,
            content=content,
            metadata=json.loads(metadata_json),
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

