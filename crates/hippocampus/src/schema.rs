/// Optimal page size for RelaxedIdbVFS (fewer IndexedDB writes per SQLite page).
pub const PRAGMA_PAGE_SIZE: &str = "PRAGMA page_size = 65536";

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

/// All DDL statements in initialization order.
pub const SCHEMA_DDL: &[&str] = &[
    PRAGMA_PAGE_SIZE,
    CREATE_MEMORY_OBJECTS,
    CREATE_MEMORY_FTS,
    CREATE_FTS_TRIGGER_INSERT,
    CREATE_FTS_TRIGGER_DELETE,
    CREATE_FTS_TRIGGER_UPDATE,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_strings_non_empty() {
        for ddl in SCHEMA_DDL {
            assert!(!ddl.trim().is_empty(), "DDL statement is empty");
        }
    }

    #[test]
    fn schema_has_fts5() {
        assert!(CREATE_MEMORY_FTS.contains("fts5"));
    }

    #[test]
    fn schema_has_triggers() {
        assert!(CREATE_FTS_TRIGGER_INSERT.contains("AFTER INSERT"));
        assert!(CREATE_FTS_TRIGGER_DELETE.contains("AFTER DELETE"));
        assert!(CREATE_FTS_TRIGGER_UPDATE.contains("AFTER UPDATE"));
    }
}
