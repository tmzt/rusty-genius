use thiserror::Error;

#[derive(Debug, Error)]
pub enum GyrusError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] sqlx::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Memory error: {0}")]
    Memory(String),
}
