"""Search functions for full-text and semantic retrieval."""

import os
import re
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from bartleby.lib.embeddings import embed_chunk


# Pre-compiled regex for detecting FTS5 operators
_FTS_OPERATOR_RE = re.compile(r'\b(AND|OR|NOT)\b|"')


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5. Splits into words, strips special chars.

    If the user already included FTS5 operators (AND, OR, NOT, quotes),
    the query is passed through unchanged.  Otherwise each word is cleaned
    of FTS5 special characters and returned as implicit-AND terms.
    """
    if _FTS_OPERATOR_RE.search(query):
        return query
    # Quote each term so FTS5's own tokenizer handles punctuation
    # correctly (e.g. "PM2.5" tokenizes to the phrase "pm2 5").
    words = query.split()
    return ' '.join(f'"{w}"' for w in words if w)


class SearchResult:
    """A search result containing chunk information and metadata."""

    def __init__(
        self,
        chunk_id: str,
        body: str,
        score: float,
        page_number: Optional[int] = None,
        document_id: Optional[str] = None,
        origin_file_path: Optional[str] = None,
        chunk_index: Optional[int] = None,
        section_heading: Optional[str] = None,
        content_type: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.body = body
        self.score = score
        self.page_number = page_number
        self.document_id = document_id
        self.origin_file_path = origin_file_path
        self.chunk_index = chunk_index
        self.section_heading = section_heading
        self.content_type = content_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "body": self.body,
            "score": self.score,
            "page_number": self.page_number,
            "document_id": self.document_id,
            "origin_file_path": self.origin_file_path,
            "chunk_index": self.chunk_index,
            "section_heading": self.section_heading,
            "content_type": self.content_type,
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return dict representation without the potentially large body field."""
        data = self.to_dict()
        data.pop("body", None)
        return data

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, page={self.page_number}, doc={self.document_id})"


def full_text_search(
    connection,
    query: str,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> List[SearchResult]:
    """
    Perform full-text search using SQLite FTS5.

    Args:
        connection: APSW database connection
        query: Search query string (supports FTS5 query syntax)
        limit: Maximum number of results to return
        document_id: Optional filter to search within a specific document

    Returns:
        List of SearchResult objects ranked by relevance
    """
    cursor = connection.cursor()

    sanitized_query = _sanitize_fts_query(query)

    sql = """
        SELECT
            c.chunk_id,
            c.body,
            fts.rank as score,
            p.page_number,
            d.document_id,
            d.origin_file_path,
            c.chunk_index,
            c.section_heading,
            c.content_type
        FROM fts_chunks fts
        JOIN chunks c ON fts.chunk_id = c.chunk_id
        JOIN pages p ON c.page_id = p.page_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE fts_chunks MATCH ?
    """
    params = [sanitized_query]
    if document_id:
        sql += "\n          AND d.document_id = ?"
        params.append(document_id)
    sql += """
        ORDER BY rank
        LIMIT ?
    """

    results = []
    params.append(limit)
    for row in cursor.execute(sql, params):
        results.append(SearchResult(
            chunk_id=row[0],
            body=row[1],
            score=abs(row[2]),  # FTS5 rank is negative, lower is better
            page_number=row[3],
            document_id=row[4],
            origin_file_path=row[5],
            chunk_index=row[6],
            section_heading=row[7],
            content_type=row[8],
        ))

    return results


def semantic_search(
    connection,
    query: str,
    embedding_model: SentenceTransformer,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> List[SearchResult]:
    """
    Perform semantic search using vector similarity.

    Args:
        connection: APSW database connection
        query: Search query string
        embedding_model: SentenceTransformer model for generating embeddings
        limit: Maximum number of results to return
        document_id: Optional filter to search within a specific document

    Returns:
        List of SearchResult objects ranked by cosine similarity
    """
    cursor = connection.cursor()

    query_embedding = embed_chunk(embedding_model, query)
    query_bytes = query_embedding.astype("float32").tobytes()

    sql = """
        SELECT
            c.chunk_id,
            c.body,
            vec_distance_cosine(vc.embedding, ?) as distance,
            p.page_number,
            d.document_id,
            d.origin_file_path,
            c.chunk_index,
            c.section_heading,
            c.content_type
        FROM vec_chunks vc
        JOIN chunks c ON vc.chunk_id = c.chunk_id
        JOIN pages p ON c.page_id = p.page_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE vc.embedding MATCH ?
          AND k = ?
    """
    params = [query_bytes, query_bytes, limit]
    if document_id:
        sql += "\n          AND d.document_id = ?"
        params.append(document_id)
    sql += """
        ORDER BY distance
    """

    results = []
    for row in cursor.execute(sql, params):
        similarity = 1.0 - row[2]
        results.append(SearchResult(
            chunk_id=row[0],
            body=row[1],
            score=similarity,
            page_number=row[3],
            document_id=row[4],
            origin_file_path=row[5],
            chunk_index=row[6],
            section_heading=row[7],
            content_type=row[8],
        ))

    return results


def get_document_chunks(
    connection,
    document_id: str,
    start_chunk: int = 0,
    max_chunks: Optional[int] = None,
) -> List[SearchResult]:
    """
    Retrieve chunks for a specific document.

    Args:
        connection: APSW database connection
        document_id: Document ID to retrieve chunks for
        start_chunk: Zero-based starting chunk offset within the document
        max_chunks: Maximum number of chunks to return (None = all)

    Returns:
        List of SearchResult objects ordered by page and chunk index
    """
    cursor = connection.cursor()

    sql = """
        SELECT
            c.chunk_id,
            c.body,
            c.chunk_index,
            p.page_number,
            d.document_id,
            d.origin_file_path,
            c.section_heading,
            c.content_type
        FROM chunks c
        JOIN pages p ON c.page_id = p.page_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE d.document_id = ?
        ORDER BY p.page_number, c.chunk_index
    """

    params = [document_id]
    if max_chunks is not None:
        offset = max(0, start_chunk)
        limit = max(0, max_chunks)
        sql += "\n        LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    results = []
    for row in cursor.execute(sql, params):
        results.append(SearchResult(
            chunk_id=row[0],
            body=row[1],
            score=1.0,  # Not ranked
            page_number=row[3],
            document_id=row[4],
            origin_file_path=row[5],
            chunk_index=row[2],
            section_heading=row[6],
            content_type=row[7],
        ))

    return results


def count_document_chunks(
    connection,
    document_id: str,
) -> int:
    """
    Count total chunks for a specific document.

    Args:
        connection: APSW database connection
        document_id: Document ID to count chunks for

    Returns:
        Total number of chunks for the document
    """
    cursor = connection.cursor()

    sql = """
        SELECT COUNT(*)
        FROM chunks c
        JOIN pages p ON c.page_id = p.page_id
        WHERE p.document_id = ?
    """

    cursor.execute(sql, (document_id,))
    row = cursor.fetchone()
    return row[0] if row else 0


def get_chunk_window_by_chunk_id(
    connection,
    chunk_id: str,
    window_radius: int,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a small window of chunks around a specific chunk ID.

    Args:
        connection: APSW database connection
        chunk_id: Anchor chunk ID
        window_radius: Number of chunks to include before/after the anchor

    Returns:
        Dictionary with window metadata and chunk data, or None if chunk not found
    """
    cursor = connection.cursor()

    sql = """
        SELECT
            c.chunk_index,
            d.document_id,
            d.origin_file_path
        FROM chunks c
        JOIN pages p ON c.page_id = p.page_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE c.chunk_id = ?
    """

    cursor.execute(sql, (chunk_id,))
    row = cursor.fetchone()

    if not row:
        return None

    chunk_index, document_id, origin_file_path = row
    if chunk_index is None or document_id is None:
        return None

    start_chunk = max(0, chunk_index - window_radius)
    max_chunks = window_radius * 2 + 1
    chunks = get_document_chunks(connection, document_id, start_chunk=start_chunk, max_chunks=max_chunks)

    return {
        "document_id": document_id,
        "origin_file_path": origin_file_path,
        "center_chunk_index": chunk_index,
        "start_chunk": start_chunk,
        "returned_chunks": len(chunks),
        "max_chunks": max_chunks,
        "chunks": [c.to_dict() for c in chunks],
    }


# --- Library query functions ---


def list_all_documents(connection) -> List[Dict[str, Any]]:
    """List all documents with metadata and summary info."""
    cursor = connection.cursor()

    sql = """
        SELECT
            d.document_id,
            d.origin_file_path,
            d.pages_count,
            COUNT(DISTINCT c.chunk_id) as chunk_count,
            s.title,
            s.subtitle,
            CASE WHEN s.document_id IS NOT NULL THEN 1 ELSE 0 END as has_summary
        FROM documents d
        LEFT JOIN pages p ON p.document_id = d.document_id
        LEFT JOIN chunks c ON c.page_id = p.page_id
        LEFT JOIN summaries s ON s.document_id = d.document_id
        GROUP BY d.document_id
        ORDER BY d.origin_file_path
    """

    results = []
    for row in cursor.execute(sql):
        origin = row[1] or ""
        filename = os.path.splitext(os.path.basename(origin))[0] if origin else ""
        results.append({
            "document_id": row[0],
            "filename": filename,
            "origin_file_path": origin,
            "pages_count": row[2],
            "chunk_count": row[3],
            "title": row[4],
            "subtitle": row[5],
            "has_summary": bool(row[6]),
        })

    return results


def get_document_summary(connection, document_id: str) -> Optional[Dict[str, Any]]:
    """Get a document's summary from the summaries table.

    Returns:
        Dict with title, subtitle, body or None if no summary exists.
    """
    cursor = connection.cursor()
    cursor.execute(
        "SELECT title, subtitle, body FROM summaries WHERE document_id = ?",
        (document_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"title": row[0], "subtitle": row[1], "body": row[2]}


def save_document_summary(
    connection,
    document_id: str,
    title: str,
    subtitle: Optional[str],
    body: str,
):
    """Insert or replace a document summary."""
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO summaries (document_id, title, subtitle, body)
        VALUES (?, ?, ?, ?)
        """,
        (document_id, title, subtitle, body),
    )


def document_exists(connection, document_id: str) -> bool:
    """Check if a document exists in the database."""
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
        (document_id,),
    )
    return cursor.fetchone() is not None
