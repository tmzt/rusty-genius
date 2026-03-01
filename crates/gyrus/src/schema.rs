use sqlx::SqlitePool;

pub const CREATE_MEMORY_OBJECTS: &str = r#"
CREATE TABLE IF NOT EXISTS memory_objects (
    id TEXT PRIMARY KEY,
    short_name TEXT NOT NULL,
    long_name TEXT NOT NULL,
    description TEXT NOT NULL,
    object_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)
"#;

pub const CREATE_MEMORY_FTS: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    short_name, long_name, description, content,
    content=memory_objects, content_rowid=rowid
)
"#;

pub const CREATE_FTS_TRIGGER_INSERT: &str = r#"
CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory_objects BEGIN
  INSERT INTO memory_fts(rowid, short_name, long_name, description, content)
  VALUES (new.rowid, new.short_name, new.long_name, new.description, new.content);
END
"#;

pub const CREATE_FTS_TRIGGER_DELETE: &str = r#"
CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory_objects BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, short_name, long_name, description, content)
  VALUES ('delete', old.rowid, old.short_name, old.long_name, old.description, old.content);
END
"#;

pub const CREATE_FTS_TRIGGER_UPDATE: &str = r#"
CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory_objects BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, short_name, long_name, description, content)
  VALUES ('delete', old.rowid, old.short_name, old.long_name, old.description, old.content);
  INSERT INTO memory_fts(rowid, short_name, long_name, description, content)
  VALUES (new.rowid, new.short_name, new.long_name, new.description, new.content);
END
"#;

/// vec0 virtual table (requires sqlite-vec extension)
#[cfg(feature = "vec0")]
pub const CREATE_MEMORY_VEC: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[384]
)
"#;

/// Fallback: regular table storing embeddings as JSON blobs
#[cfg(not(feature = "vec0"))]
pub const CREATE_MEMORY_EMBEDDINGS: &str = r#"
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,
    FOREIGN KEY (id) REFERENCES memory_objects(id) ON DELETE CASCADE
)
"#;

/// Initialize the database schema: base table, FTS5, triggers, and vector storage.
pub async fn init_db(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query("PRAGMA foreign_keys = ON").execute(pool).await?;

    sqlx::query(CREATE_MEMORY_OBJECTS).execute(pool).await?;
    sqlx::query(CREATE_MEMORY_FTS).execute(pool).await?;
    sqlx::query(CREATE_FTS_TRIGGER_INSERT).execute(pool).await?;
    sqlx::query(CREATE_FTS_TRIGGER_DELETE).execute(pool).await?;
    sqlx::query(CREATE_FTS_TRIGGER_UPDATE).execute(pool).await?;

    #[cfg(feature = "vec0")]
    sqlx::query(CREATE_MEMORY_VEC).execute(pool).await?;

    #[cfg(not(feature = "vec0"))]
    sqlx::query(CREATE_MEMORY_EMBEDDINGS).execute(pool).await?;

    Ok(())
}
