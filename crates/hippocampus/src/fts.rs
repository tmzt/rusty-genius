use sqlite_wasm_rs::export::install_relaxed_idb;
use sqlite_wasm_rs::ffi;
use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::ptr;

use crate::schema::SCHEMA_DDL;

/// FTS5-backed full-text index stored in IndexedDB via sqlite-wasm-rs RelaxedIdbVFS.
pub struct FtsIndex {
    db: *mut ffi::sqlite3,
}

impl FtsIndex {
    /// Open (or create) the SQLite database with the given name.
    /// The RelaxedIdbVFS persists pages to IndexedDB automatically.
    pub async fn open(db_name: &str) -> Result<Self, String> {
        // Install the IndexedDB-backed VFS (idempotent)
        install_relaxed_idb().await;

        let c_name = CString::new(db_name).map_err(|e| e.to_string())?;
        let mut db: *mut ffi::sqlite3 = ptr::null_mut();

        let rc = unsafe {
            ffi::sqlite3_open_v2(
                c_name.as_ptr(),
                &mut db,
                ffi::SQLITE_OPEN_READWRITE | ffi::SQLITE_OPEN_CREATE,
                ptr::null(), // uses default (RelaxedIdb) VFS
            )
        };

        if rc != ffi::SQLITE_OK {
            let msg = unsafe {
                let raw = ffi::sqlite3_errmsg(db);
                if raw.is_null() {
                    "unknown error".to_string()
                } else {
                    CStr::from_ptr(raw).to_string_lossy().into_owned()
                }
            };
            return Err(format!("sqlite3_open_v2 failed ({}): {}", rc, msg));
        }

        let index = Self { db };

        // Initialize schema
        for ddl in SCHEMA_DDL {
            index.exec(ddl)?;
        }

        Ok(index)
    }

    /// Index (upsert) a memory object into the SQLite table.
    /// FTS5 triggers keep the full-text index in sync automatically.
    pub fn index_object(
        &self,
        id: &str,
        short_name: &str,
        long_name: &str,
        description: &str,
        object_type: &str,
        content: &str,
        metadata: Option<&str>,
        created_at: u64,
        updated_at: u64,
    ) -> Result<(), String> {
        let sql = r#"
            INSERT OR REPLACE INTO memory_objects
                (id, short_name, long_name, description, object_type, content, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;

        let stmt = self.prepare(sql)?;

        self.bind_text(stmt, 1, id)?;
        self.bind_text(stmt, 2, short_name)?;
        self.bind_text(stmt, 3, long_name)?;
        self.bind_text(stmt, 4, description)?;
        self.bind_text(stmt, 5, object_type)?;
        self.bind_text(stmt, 6, content)?;
        match metadata {
            Some(m) => self.bind_text(stmt, 7, m)?,
            None => {
                let rc = unsafe { ffi::sqlite3_bind_null(stmt, 7) };
                if rc != ffi::SQLITE_OK {
                    self.finalize(stmt);
                    return Err(format!("bind_null failed: {}", rc));
                }
            }
        }
        self.bind_int64(stmt, 8, created_at as i64)?;
        self.bind_int64(stmt, 9, updated_at as i64)?;

        let rc = unsafe { ffi::sqlite3_step(stmt) };
        self.finalize(stmt);

        if rc != ffi::SQLITE_DONE {
            return Err(format!("index_object step failed: {}", rc));
        }
        Ok(())
    }

    /// Remove a memory object from the SQLite table (triggers clean up FTS).
    pub fn remove_object(&self, id: &str) -> Result<(), String> {
        let sql = "DELETE FROM memory_objects WHERE id = ?1";
        let stmt = self.prepare(sql)?;
        self.bind_text(stmt, 1, id)?;

        let rc = unsafe { ffi::sqlite3_step(stmt) };
        self.finalize(stmt);

        if rc != ffi::SQLITE_DONE {
            return Err(format!("remove_object step failed: {}", rc));
        }
        Ok(())
    }

    /// Full-text search returning matching IDs ranked by FTS5 relevance.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<String>, String> {
        let sql = r#"
            SELECT mo.id
            FROM memory_fts
            JOIN memory_objects mo ON memory_fts.rowid = mo.rowid
            WHERE memory_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
        "#;

        let stmt = self.prepare(sql)?;
        self.bind_text(stmt, 1, query)?;
        self.bind_int64(stmt, 2, limit as i64)?;

        let mut ids = Vec::new();
        loop {
            let rc = unsafe { ffi::sqlite3_step(stmt) };
            if rc == ffi::SQLITE_ROW {
                let raw = unsafe { ffi::sqlite3_column_text(stmt, 0) };
                if !raw.is_null() {
                    let s = unsafe { CStr::from_ptr(raw as *const _) }
                        .to_string_lossy()
                        .into_owned();
                    ids.push(s);
                }
            } else if rc == ffi::SQLITE_DONE {
                break;
            } else {
                self.finalize(stmt);
                return Err(format!("search step failed: {}", rc));
            }
        }
        self.finalize(stmt);
        Ok(ids)
    }

    /// List all object IDs matching a given object_type JSON string.
    pub fn list_by_type(&self, object_type_json: &str) -> Result<Vec<String>, String> {
        let sql = "SELECT id FROM memory_objects WHERE object_type = ?1";
        let stmt = self.prepare(sql)?;
        self.bind_text(stmt, 1, object_type_json)?;

        let mut ids = Vec::new();
        loop {
            let rc = unsafe { ffi::sqlite3_step(stmt) };
            if rc == ffi::SQLITE_ROW {
                let raw = unsafe { ffi::sqlite3_column_text(stmt, 0) };
                if !raw.is_null() {
                    let s = unsafe { CStr::from_ptr(raw as *const _) }
                        .to_string_lossy()
                        .into_owned();
                    ids.push(s);
                }
            } else if rc == ffi::SQLITE_DONE {
                break;
            } else {
                self.finalize(stmt);
                return Err(format!("list_by_type step failed: {}", rc));
            }
        }
        self.finalize(stmt);
        Ok(ids)
    }

    /// List all object IDs in the table.
    pub fn list_all_ids(&self) -> Result<Vec<String>, String> {
        let sql = "SELECT id FROM memory_objects";
        let stmt = self.prepare(sql)?;

        let mut ids = Vec::new();
        loop {
            let rc = unsafe { ffi::sqlite3_step(stmt) };
            if rc == ffi::SQLITE_ROW {
                let raw = unsafe { ffi::sqlite3_column_text(stmt, 0) };
                if !raw.is_null() {
                    let s = unsafe { CStr::from_ptr(raw as *const _) }
                        .to_string_lossy()
                        .into_owned();
                    ids.push(s);
                }
            } else if rc == ffi::SQLITE_DONE {
                break;
            } else {
                self.finalize(stmt);
                return Err(format!("list_all_ids step failed: {}", rc));
            }
        }
        self.finalize(stmt);
        Ok(ids)
    }

    /// Drop all data from the tables.
    pub fn clear(&self) -> Result<(), String> {
        // Delete from memory_objects triggers FTS cleanup
        self.exec("DELETE FROM memory_objects")?;
        Ok(())
    }

    // ── Helpers ──

    fn exec(&self, sql: &str) -> Result<(), String> {
        let c_sql = CString::new(sql).map_err(|e| e.to_string())?;
        let rc = unsafe {
            ffi::sqlite3_exec(
                self.db,
                c_sql.as_ptr(),
                None,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if rc != ffi::SQLITE_OK {
            Err(format!("exec failed ({}): {}", rc, sql.chars().take(60).collect::<String>()))
        } else {
            Ok(())
        }
    }

    fn prepare(&self, sql: &str) -> Result<*mut ffi::sqlite3_stmt, String> {
        let c_sql = CString::new(sql).map_err(|e| e.to_string())?;
        let mut stmt: *mut ffi::sqlite3_stmt = ptr::null_mut();
        let rc = unsafe {
            ffi::sqlite3_prepare_v2(
                self.db,
                c_sql.as_ptr(),
                -1,
                &mut stmt,
                ptr::null_mut(),
            )
        };
        if rc != ffi::SQLITE_OK {
            Err(format!("prepare failed ({}): {}", rc, sql.chars().take(60).collect::<String>()))
        } else {
            Ok(stmt)
        }
    }

    fn bind_text(&self, stmt: *mut ffi::sqlite3_stmt, idx: c_int, val: &str) -> Result<(), String> {
        let c_val = CString::new(val).map_err(|e| e.to_string())?;
        let rc = unsafe {
            ffi::sqlite3_bind_text(
                stmt,
                idx,
                c_val.as_ptr(),
                val.len() as c_int,
                ffi::SQLITE_TRANSIENT(),
            )
        };
        if rc != ffi::SQLITE_OK {
            self.finalize(stmt);
            Err(format!("bind_text failed at idx {}: {}", idx, rc))
        } else {
            Ok(())
        }
    }

    fn bind_int64(&self, stmt: *mut ffi::sqlite3_stmt, idx: c_int, val: i64) -> Result<(), String> {
        let rc = unsafe { ffi::sqlite3_bind_int64(stmt, idx, val) };
        if rc != ffi::SQLITE_OK {
            self.finalize(stmt);
            Err(format!("bind_int64 failed at idx {}: {}", idx, rc))
        } else {
            Ok(())
        }
    }

    fn finalize(&self, stmt: *mut ffi::sqlite3_stmt) {
        unsafe {
            ffi::sqlite3_finalize(stmt);
        }
    }
}

impl Drop for FtsIndex {
    fn drop(&mut self) {
        if !self.db.is_null() {
            unsafe {
                ffi::sqlite3_close_v2(self.db);
            }
        }
    }
}
