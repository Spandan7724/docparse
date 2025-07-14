use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextExtractionConfig {
    pub line_clustering_tolerance: f64,
    pub space_threshold_height: f64,
    pub space_threshold_width: f64,
    pub min_character_size: f64,
}

impl Default for TextExtractionConfig {
    fn default() -> Self {
        Self {
            line_clustering_tolerance: 0.25,
            space_threshold_height: 0.20,
            space_threshold_width: 0.40,
            min_character_size: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingConfig {
    pub default_dpi: f64,
    pub points_per_inch: f64,
    pub auto_rotate_landscape: bool,
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            default_dpi: 224.0,
            points_per_inch: 72.0,
            auto_rotate_landscape: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub pdfium_library_path: String,
    pub models_directory: String,
    pub cache_directory: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            pdfium_library_path: "./".to_string(),
            models_directory: "models/".to_string(),
            cache_directory: "".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocparseConfig {
    pub text_extraction: TextExtractionConfig,
    pub rendering: RenderingConfig,
    pub paths: PathsConfig,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub overrides: HashMap<String, toml::Value>,
}

impl Default for DocparseConfig {
    fn default() -> Self {
        Self {
            text_extraction: TextExtractionConfig::default(),
            rendering: RenderingConfig::default(),
            paths: PathsConfig::default(),
            profiles: HashMap::new(),
        }
    }
}

impl DocparseConfig {
    /// Load configuration from file with environment variable overrides
    pub fn load() -> anyhow::Result<Self> {
        let mut config = Self::load_from_file()?;
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file
    pub fn load_from_file() -> anyhow::Result<Self> {
        // Try to find config file in order of preference
        let config_paths = [
            "config.toml",
            "./config.toml", 
            "docparse.toml",
            "./docparse.toml",
        ];

        for path in &config_paths {
            if Path::new(path).exists() {
                let content = fs::read_to_string(path)?;
                let config: DocparseConfig = toml::from_str(&content)?;
                return Ok(config);
            }
        }

        // Return default config if no file found
        Ok(DocparseConfig::default())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) -> anyhow::Result<()> {
        // Text extraction overrides
        if let Ok(val) = env::var("DOCPARSE_LINE_CLUSTERING_TOLERANCE") {
            self.text_extraction.line_clustering_tolerance = val.parse()?;
        }
        if let Ok(val) = env::var("DOCPARSE_SPACE_THRESHOLD_HEIGHT") {
            self.text_extraction.space_threshold_height = val.parse()?;
        }
        if let Ok(val) = env::var("DOCPARSE_SPACE_THRESHOLD_WIDTH") {
            self.text_extraction.space_threshold_width = val.parse()?;
        }
        if let Ok(val) = env::var("DOCPARSE_MIN_CHARACTER_SIZE") {
            self.text_extraction.min_character_size = val.parse()?;
        }

        // Rendering overrides
        if let Ok(val) = env::var("DOCPARSE_DEFAULT_DPI") {
            self.rendering.default_dpi = val.parse()?;
        }
        if let Ok(val) = env::var("DOCPARSE_POINTS_PER_INCH") {
            self.rendering.points_per_inch = val.parse()?;
        }
        if let Ok(val) = env::var("DOCPARSE_AUTO_ROTATE_LANDSCAPE") {
            self.rendering.auto_rotate_landscape = val.parse()?;
        }

        // Paths overrides
        if let Ok(val) = env::var("DOCPARSE_PDFIUM_LIBRARY_PATH") {
            self.paths.pdfium_library_path = val;
        }
        if let Ok(val) = env::var("DOCPARSE_MODELS_DIRECTORY") {
            self.paths.models_directory = val;
        }
        if let Ok(val) = env::var("DOCPARSE_CACHE_DIRECTORY") {
            self.paths.cache_directory = val;
        }

        Ok(())
    }

    /// Apply a profile configuration
    pub fn apply_profile(&mut self, profile_name: &str) -> anyhow::Result<()> {
        if let Some(profile) = self.profiles.get(profile_name).cloned() {
            // Apply profile overrides to current config
            for (key, value) in profile.overrides {
                self.apply_override(&key, &value)?;
            }
        }
        Ok(())
    }

    /// Apply a single configuration override
    fn apply_override(&mut self, key: &str, value: &toml::Value) -> anyhow::Result<()> {
        match key {
            "line_clustering_tolerance" => {
                if let Some(val) = value.as_float() {
                    self.text_extraction.line_clustering_tolerance = val;
                }
            }
            "space_threshold_height" => {
                if let Some(val) = value.as_float() {
                    self.text_extraction.space_threshold_height = val;
                }
            }
            "space_threshold_width" => {
                if let Some(val) = value.as_float() {
                    self.text_extraction.space_threshold_width = val;
                }
            }
            "default_dpi" | "dpi" => {
                if let Some(val) = value.as_float() {
                    self.rendering.default_dpi = val;
                }
            }
            "confidence_threshold" => {
                // This will be handled by Python config
            }
            _ => {
                // Ignore unknown keys for forward compatibility
            }
        }
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate DPI ranges
        if self.rendering.default_dpi < 72.0 || self.rendering.default_dpi > 600.0 {
            return Err(anyhow::anyhow!(
                "Default DPI {} is outside valid range 72-600",
                self.rendering.default_dpi
            ));
        }
        
        // Validate text extraction parameters
        if self.text_extraction.line_clustering_tolerance < 0.0 || 
           self.text_extraction.line_clustering_tolerance > 1.0 {
            return Err(anyhow::anyhow!(
                "Line clustering tolerance {} must be between 0.0 and 1.0",
                self.text_extraction.line_clustering_tolerance
            ));
        }
        
        if self.text_extraction.space_threshold_height < 0.0 || 
           self.text_extraction.space_threshold_height > 1.0 {
            return Err(anyhow::anyhow!(
                "Space threshold height {} must be between 0.0 and 1.0",
                self.text_extraction.space_threshold_height
            ));
        }
        
        if self.text_extraction.space_threshold_width < 0.0 || 
           self.text_extraction.space_threshold_width > 1.0 {
            return Err(anyhow::anyhow!(
                "Space threshold width {} must be between 0.0 and 1.0",
                self.text_extraction.space_threshold_width
            ));
        }
        
        if self.text_extraction.min_character_size < 0.0 {
            return Err(anyhow::anyhow!(
                "Minimum character size {} cannot be negative",
                self.text_extraction.min_character_size
            ));
        }
        
        // Validate points per inch
        if self.rendering.points_per_inch <= 0.0 {
            return Err(anyhow::anyhow!(
                "Points per inch {} must be positive",
                self.rendering.points_per_inch
            ));
        }
        
        Ok(())
    }
}

/// Global configuration instance
use once_cell::sync::OnceCell;
static CONFIG: OnceCell<DocparseConfig> = OnceCell::new();

/// Get the global configuration instance
pub fn get_config() -> &'static DocparseConfig {
    CONFIG.get_or_init(|| {
        DocparseConfig::load().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to load config: {}. Using defaults.", e);
            DocparseConfig::default()
        })
    })
}

/// Initialize configuration with optional profile
pub fn init_config(profile: Option<&str>) -> anyhow::Result<()> {
    let mut config = DocparseConfig::load()?;
    
    if let Some(profile_name) = profile {
        config.apply_profile(profile_name)?;
    }
    
    // Validate again after profile application
    config.validate()?;
    
    CONFIG.set(config).map_err(|_| {
        anyhow::anyhow!("Configuration already initialized")
    })?;
    
    Ok(())
}