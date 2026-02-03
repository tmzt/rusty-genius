use thiserror::Error;

#[derive(Error, Debug)]
pub enum GeniusError {
    #[error("Protocol Error: {0}")]
    ProtocolError(String),

    #[error("Manifest Error: {0}")]
    ManifestError(String),

    #[error("Asset Error: {0}")]
    AssetError(String),

    #[error("Engine Error: {0}")]
    EngineError(String),

    #[error("Unknown Error: {0}")]
    Unknown(String),
}
