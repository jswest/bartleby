"""Search functions for full-text and semantic retrieval."""

from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from bartleby.lib.embeddings import embed_chunk
from bartleby.read.sqlite import get_connection


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
    ):
        self.chunk_id = chunk_id
        self.body = body
        self.score = score
        self.page_number = page_number
        self.document_id = document_id
        self.origin_file_path = origin_file_path
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "body": self.body,
            "score": self.score,
            "page_number": self.page_number,
            "document_id": self.document_id,
            "origin_file_path": self.origin_file_path,
            "chunk_index": self.chunk_index,
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return dict representation without the potentially large body field."""
        data = self.to_dict()
        data.pop("body", None)
        return data

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, page={self.page_number}, doc={self.document_id})"


def full_text_search(
    db_path: Path,
    query: str,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> List[SearchResult]:
    """
    Perform full-text search using SQLite FTS5.

    Args:
        db_path: Path to the database file
        query: Search query string (supports FTS5 query syntax)
        limit: Maximum number of results to return

    Returns:
        List of SearchResult objects ranked by relevance
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    # Sanitize query for FTS5 - wrap in quotes to handle special characters
    # FTS5 doesn't like periods, colons, and other special chars in unquoted queries
    sanitized_query = f'"{query}"'

    sql = """
        SELECT
            c.chunk_id,
            c.body,
            fts.rank as score,
            p.page_number,
            d.document_id,
            d.origin_file_path,
            c.chunk_index
        FROM fts_chunks fts
        JOIN chunks c ON fts.chunk_id = c.chunk_id
        LEFT JOIN pages p ON c.page_id = p.page_id
        LEFT JOIN summaries s ON c.summary_id = s.summary_id
        LEFT JOIN pages sp ON s.page_id = sp.page_id
        LEFT JOIN documents d ON COALESCE(p.document_id, sp.document_id) = d.document_id
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
    try:
        params.append(limit)
        for row in cursor.execute(sql, params):
            results.append(SearchResult(
                chunk_id=row[0],
                body=row[1],
                score=abs(row[2]), # FTS5 rank is negative, lower is better
                page_number=row[3],
                document_id=row[4],
                origin_file_path=row[5],
                chunk_index=row[6],
            ))
    except Exception:
        # If FTS5 query fails, return empty results
        pass

    connection.close()
    return results


def semantic_search(
    db_path: Path,
    query: str,
    embedding_model: SentenceTransformer,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> List[SearchResult]:
    """
    Perform semantic search using vector similarity.

    Args:
        db_path: Path to the database file
        query: Search query string
        embedding_model: SentenceTransformer model for generating embeddings
        limit: Maximum number of results to return

    Returns:
        List of SearchResult objects ranked by cosine similarity
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    # Generate query embedding
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
            c.chunk_index
        FROM vec_chunks vc
        JOIN chunks c ON vc.chunk_id = c.chunk_id
        LEFT JOIN pages p ON c.page_id = p.page_id
        LEFT JOIN summaries s ON c.summary_id = s.summary_id
        LEFT JOIN pages sp ON s.page_id = sp.page_id
        LEFT JOIN documents d ON COALESCE(p.document_id, sp.document_id) = d.document_id
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
        # Convert distance to similarity score (1 - distance)
        similarity = 1.0 - row[2]
        results.append(SearchResult(
            chunk_id=row[0],
            body=row[1],
            score=similarity,
            page_number=row[3],
            document_id=row[4],
            origin_file_path=row[5],
            chunk_index=row[6],
        ))

    connection.close()
    return results


def get_document_chunks(
    db_path: Path,
    document_id: str,
    start_chunk: int = 0,
    max_chunks: Optional[int] = None,
) -> List[SearchResult]:
    """
    Retrieve chunks for a specific document.

    Args:
        db_path: Path to the database file
        document_id: Document ID to retrieve chunks for
        start_chunk: Zero-based starting chunk offset within the document
        max_chunks: Maximum number of chunks to return (None = all)

    Returns:
        List of SearchResult objects ordered by page and chunk index
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    sql = """
        SELECT
            c.chunk_id,
            c.body,
            c.chunk_index,
            p.page_number,
            d.document_id,
            d.origin_file_path
        FROM chunks c
        LEFT JOIN pages p ON c.page_id = p.page_id
        LEFT JOIN summaries s ON c.summary_id = s.summary_id
        LEFT JOIN pages sp ON s.page_id = sp.page_id
        LEFT JOIN documents d ON COALESCE(p.document_id, sp.document_id) = d.document_id
        WHERE d.document_id = ?
        ORDER BY COALESCE(p.page_number, sp.page_number), c.chunk_index
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
        ))

    connection.close()
    return results


def count_document_chunks(
    db_path: Path,
    document_id: str,
) -> int:
    """
    Count total chunks for a specific document.

    Args:
        db_path: Path to the database file
        document_id: Document ID to count chunks for

    Returns:
        Total number of chunks for the document
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    sql = """
        SELECT COUNT(*)
        FROM chunks c
        LEFT JOIN pages p ON c.page_id = p.page_id
        LEFT JOIN summaries s ON c.summary_id = s.summary_id
        LEFT JOIN pages sp ON s.page_id = sp.page_id
        LEFT JOIN documents d ON COALESCE(p.document_id, sp.document_id) = d.document_id
        WHERE d.document_id = ?
    """

    cursor.execute(sql, (document_id,))
    row = cursor.fetchone()
    connection.close()
    return row[0] if row else 0


def list_documents(db_path: Path) -> List[Dict[str, Any]]:
    """
    List all documents in the database.

    Args:
        db_path: Path to the database file

    Returns:
        List of dictionaries containing document metadata
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    sql = """
        SELECT
            document_id,
            origin_file_path,
            pages_count
        FROM documents
        ORDER BY origin_file_path
    """

    results = []
    for row in cursor.execute(sql):
        results.append({
            "document_id": row[0],
            "origin_file_path": row[1],
            "pages_count": row[2],
        })

    connection.close()
    return results


def get_chunk_window_by_chunk_id(
    db_path: Path,
    chunk_id: str,
    window_radius: int,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a small window of chunks around a specific chunk ID.

    Args:
        db_path: Path to the database file
        chunk_id: Anchor chunk ID
        window_radius: Number of chunks to include before/after the anchor

    Returns:
        Dictionary with window metadata and chunk data, or None if chunk not found
    """
    connection = get_connection(db_path)
    cursor = connection.cursor()

    sql = """
        SELECT
            c.chunk_index,
            d.document_id,
            d.origin_file_path
        FROM chunks c
        LEFT JOIN pages p ON c.page_id = p.page_id
        LEFT JOIN summaries s ON c.summary_id = s.summary_id
        LEFT JOIN pages sp ON s.page_id = sp.page_id
        LEFT JOIN documents d ON COALESCE(p.document_id, sp.document_id) = d.document_id
        WHERE c.chunk_id = ?
    """

    cursor.execute(sql, (chunk_id,))
    row = cursor.fetchone()
    connection.close()

    if not row:
        return None

    chunk_index, document_id, origin_file_path = row
    if chunk_index is None or document_id is None:
        return None

    start_chunk = max(0, chunk_index - window_radius)
    max_chunks = window_radius * 2 + 1
    chunks = get_document_chunks(db_path, document_id, start_chunk=start_chunk, max_chunks=max_chunks)

    return {
        "document_id": document_id,
        "origin_file_path": origin_file_path,
        "center_chunk_index": chunk_index,
        "start_chunk": start_chunk,
        "returned_chunks": len(chunks),
        "max_chunks": max_chunks,
        "chunks": [c.to_dict() for c in chunks],
    }
