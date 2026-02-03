use anyhow::{Context, Result};
use rusty_genius_core::manifest::ModelSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

const DEFAULT_MODELS: &str = include_str!("models.toml");

#[derive(Debug, Deserialize)]
struct RegistryFile {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub repo: String,
    pub filename: String,
    pub quantization: String,
}

pub struct ModelRegistry {
    config_dir: PathBuf,
    models: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let config_dir = dirs::config_dir()
            .context("Could not find config directory")?
            .join("rusty-genius");

        fs::create_dir_all(&config_dir)?;

        let mut registry = Self {
            config_dir,
            models: HashMap::new(),
        };

        registry.load_defaults()?;
        registry.load_local()?;

        Ok(registry)
    }

    fn load_defaults(&mut self) -> Result<()> {
        let parsed: RegistryFile = toml::from_str(DEFAULT_MODELS)?;
        for model in parsed.models {
            self.models.insert(model.name.clone(), model);
        }
        Ok(())
    }

    fn load_local(&mut self) -> Result<()> {
        let registry_path = self.config_dir.join("registry.toml");
        if registry_path.exists() {
            let content = fs::read_to_string(registry_path)?;
            let parsed: RegistryFile = toml::from_str(&content)?;
            for model in parsed.models {
                self.models.insert(model.name.clone(), model);
            }
        }
        Ok(())
    }

    pub fn resolve(&self, name_or_spec: &str) -> Option<ModelSpec> {
        if let Some(entry) = self.models.get(name_or_spec) {
            return Some(ModelSpec {
                repo: entry.repo.clone(),
                filename: entry.filename.clone(),
                quantization: entry.quantization.clone(),
            });
        }
        None
    }

    pub fn get_cache_dir(&self) -> PathBuf {
        self.config_dir.join("cache")
    }
}
