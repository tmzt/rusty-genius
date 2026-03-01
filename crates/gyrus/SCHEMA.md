# Gyrus Database Schema

## Tables

### `memory_objects`
Primary storage for memory objects.

```sql
CREATE TABLE memory_objects (
    id TEXT PRIMARY KEY,
    short_name TEXT NOT NULL,
    long_name TEXT NOT NULL,
    description TEXT NOT NULL,
    object_type TEXT NOT NULL,   -- application-defined type as string
    content TEXT NOT NULL,
    metadata TEXT,               -- optional JSON blob
    created_at INTEGER NOT NULL, -- unix timestamp
    updated_at INTEGER NOT NULL  -- unix timestamp
);
```

### `memory_fts` (FTS5 virtual table)
Full-text search index over memory objects, kept in sync via triggers.

```sql
CREATE VIRTUAL TABLE memory_fts USING fts5(
    short_name, long_name, description, content,
    content=memory_objects, content_rowid=rowid
);
```

### FTS5 Sync Triggers
- `memory_fts_ai` — AFTER INSERT: indexes new rows
- `memory_fts_ad` — AFTER DELETE: removes deleted rows
- `memory_fts_au` — AFTER UPDATE: re-indexes updated rows

### `memory_vec` (vec0 feature)
Vector storage using the sqlite-vec extension. Requires the `vec0` feature.

```sql
CREATE VIRTUAL TABLE memory_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[384]
);
```

### `memory_embeddings` (fallback)
Regular table storing embeddings as JSON arrays when vec0 is not available.

```sql
CREATE TABLE memory_embeddings (
    id TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,
    FOREIGN KEY (id) REFERENCES memory_objects(id) ON DELETE CASCADE
);
```

## Query Patterns

### FTS5 Search
```sql
SELECT mo.id FROM memory_fts
JOIN memory_objects mo ON memory_fts.rowid = mo.rowid
WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?;
```

### Vector Search (vec0)
```sql
SELECT id, distance FROM memory_vec
WHERE embedding MATCH ? ORDER BY distance LIMIT ?;
```

### Vector Search (fallback)
Load all embeddings and compute cosine similarity in application code.
