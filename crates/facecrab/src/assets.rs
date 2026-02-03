use crate::registry::ModelRegistry;
use anyhow::{anyhow, Context, Result};
use rusty_genius_core::manifest::ModelSpec;
use rusty_genius_core::GeniusError;
use std::fs;
use std::path::PathBuf;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

impl AssetAuthority {
    pub fn new() -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::new()?,
        })
    }

    pub fn ensure_model(&self, name: &str) -> Result<PathBuf> {
        // 1. Resolve model spec
        let spec = self.registry.resolve(name).ok_or_else(|| {
            GeniusError::ManifestError(format!("Model '{}' not found in registry", name))
        })?;

        // 2. Check cache
        let cache_dir = self.registry.get_cache_dir();
        fs::create_dir_all(&cache_dir)?;

        let path = cache_dir.join(&spec.filename);
        if path.exists() {
            return Ok(path);
        }

        // 3. Download (Simulated or Real)
        println!("Downloading {} from {}...", spec.filename, spec.repo);
        self.download_file(&spec, &path)?;

        Ok(path)
    }

    // TODO: Implement real HF download with resume capability
    // For now, this is a simplified blocking download using reqwest
    fn download_file(&self, spec: &ModelSpec, path: &PathBuf) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );

        let response = reqwest::blocking::get(&url)
            .with_context(|| format!("Failed to download from {}", url))?;

        let content = response.bytes()?;
        fs::write(path, content)?;

        Ok(())
    }
}
