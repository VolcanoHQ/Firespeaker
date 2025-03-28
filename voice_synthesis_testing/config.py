#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration management for voice synthesis testing.

This module handles loading, saving, and validating configuration settings
for the voice synthesis testing framework. It provides utilities for working
with YAML configuration files and ensures all required parameters are present.
"""

import os
import yaml
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
USER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "user.yaml")

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration dictionary.
    
    This is used when no configuration file is provided or as a base
    that's updated with user-provided configuration.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    # System detection for platform-specific defaults
    is_windows = platform.system() == "Windows"
    gpu_available = False
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not found. GPU availability cannot be determined.")
    
    # Default configuration values
    return {
        "data": {
            "sample_rate": 22050,
            "test_duration": 5,  # seconds
            "max_samples": 1000,
            "base_dir": str(Path(os.path.dirname(__file__)) / "data"),
            "raw_dir": "raw",
            "processed_dir": "processed",
            "generated_dir": "generated",
            "reference_dir": "reference"
        },
        "model": {
            "type": "transformer",
            "checkpoint_dir": str(Path(os.path.dirname(__file__)) / "models"),
            "pretrained_dir": "pretrained",
            "finetuned_dir": "finetuned",
            "checkpoints_dir": "checkpoints",
            "default_model": "base_v2.pt",
            "precision": "float32",  # or float16 for half precision
            "use_jit": True  # Use PyTorch JIT compilation
        },
        "testing": {
            "batch_size": 8 if gpu_available else 2,
            "metrics": ["mcd", "wer", "naturalness"],
            "reference_model": "reference.pt",
            "test_sentences_path": str(Path(os.path.dirname(__file__)) / "data/test_sentences.txt"),
            "results_dir": str(Path(os.path.dirname(__file__)) / "results")
        },
        "system": {
            "use_gpu": gpu_available,
            "num_workers": min(os.cpu_count() or 1, 8),  # Use at most 8 workers
            "seed": 42,
            "device": "cuda" if gpu_available else "cpu",
            "cache_dir": str(Path(os.path.dirname(__file__)) / "cache"),
            "temp_dir": str(Path(os.path.dirname(__file__)) / "temp"),
            "windows_specific": {
                "use_directml": is_windows and not gpu_available,  # DirectML acceleration for Windows
                "audio_backend": "directsound" if is_windows else "default"
            }
        },
        "visualization": {
            "dpi": 150,
            "fig_width": 12,
            "fig_height": 6,
            "colors": {
                "reference": "#1f77b4",
                "synthesized": "#ff7f0e",
                "error": "#d62728"
            },
            "export_formats": ["png", "pdf"]
        },
        "logging": {
            "level": "INFO",
            "file": str(Path(os.path.dirname(__file__)) / "logs/voice_testing.log"),
            "max_size": 10 * 1024 * 1024,  # 10 MB
            "backup_count": 5
        }
    }

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    
    Args:
        config_path: Path to the config file. If None, tries USER_CONFIG_PATH
                    then falls back to DEFAULT_CONFIG_PATH.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        ConfigurationError: If the config file doesn't exist or has invalid format.
    """
    # Start with default configuration
    config = get_default_config()
    
    # Determine which config file to load
    if config_path is None:
        if os.path.exists(USER_CONFIG_PATH):
            config_path = USER_CONFIG_PATH
            logger.info(f"Using user configuration from {USER_CONFIG_PATH}")
        elif os.path.exists(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
            logger.info(f"Using default configuration from {DEFAULT_CONFIG_PATH}")
        else:
            logger.warning("No configuration file found. Using built-in defaults.")
            return config
    elif not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    # Load and merge configuration
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            
        if not user_config:
            logger.warning(f"Empty or invalid configuration file: {config_path}")
            return config
            
        # Recursively update the default config with user values
        config = _recursive_update(config, user_config)
        logger.info(f"Configuration loaded successfully from {config_path}")
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")
        
    # Ensure required directories exist
    _ensure_directories_exist(config)
    
    return config

def _recursive_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively updates a nested dictionary with values from another dictionary.
    
    Args:
        base_dict: The base dictionary to update
        update_dict: The dictionary with updates to apply
    
    Returns:
        Dict[str, Any]: Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def _ensure_directories_exist(config: Dict[str, Any]) -> None:
    """
    Ensures all required directories in the configuration exist.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        Path(config["data"]["base_dir"]),
        Path(config["data"]["base_dir"]) / config["data"]["raw_dir"],
        Path(config["data"]["base_dir"]) / config["data"]["processed_dir"],
        Path(config["data"]["base_dir"]) / config["data"]["generated_dir"],
        Path(config["data"]["base_dir"]) / config["data"]["reference_dir"],
        Path(config["model"]["checkpoint_dir"]),
        Path(config["model"]["checkpoint_dir"]) / config["model"]["pretrained_dir"],
        Path(config["model"]["checkpoint_dir"]) / config["model"]["finetuned_dir"],
        Path(config["model"]["checkpoint_dir"]) / config["model"]["checkpoints_dir"],
        Path(config["testing"]["results_dir"]),
        Path(config["system"]["cache_dir"]),
        Path(config["system"]["temp_dir"]),
        Path(os.path.dirname(config["logging"]["file"]))
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
        except Exception as e:
            logger.warning(f"Failed to create directory {directory}: {e}")

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Saves configuration settings to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where the config will be saved
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        ConfigurationError: If the file cannot be written
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Add metadata
        config_with_meta = config.copy()
        config_with_meta["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
        
        # Write config to file
        with open(config_path, 'w') as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise ConfigurationError(f"Error saving configuration: {e}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Required top-level keys
    required_keys = ["data", "model", "testing", "system"]
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration section: {key}")
    
    # Check for valid sample rate
    if "sample_rate" in config["data"]:
        sample_rate = config["data"]["sample_rate"]
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ConfigurationError(f"Invalid sample rate: {sample_rate}")
    
    # Check for valid batch size
    if "batch_size" in config["testing"]:
        batch_size = config["testing"]["batch_size"]
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError(f"Invalid batch size: {batch_size}")
    
    # Check that model checkpoint directory exists if specified
    if "checkpoint_dir" in config["model"]:
        checkpoint_dir = Path(config["model"]["checkpoint_dir"])
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Model checkpoint directory does not exist: {checkpoint_dir}")
    
    logger.info("Configuration validation successful")
    return True

def get_test_config(base_config: Dict[str, Any], test_name: str) -> Dict[str, Any]:
    """
    Creates a test-specific configuration by adding a timestamped results directory.
    
    Args:
        base_config: Base configuration dictionary
        test_name: Name of the test
    
    Returns:
        Dict[str, Any]: Test-specific configuration
    """
    test_config = base_config.copy()
    
    # Create a timestamped directory for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(
        test_config["testing"]["results_dir"],
        f"{test_name}_{timestamp}"
    )
    
    # Update the results directory
    test_config["testing"]["current_test_dir"] = test_dir
    test_config["testing"]["current_test_name"] = test_name
    test_config["testing"]["timestamp"] = timestamp
    
    # Create the directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Save this test configuration
    save_config(test_config, os.path.join(test_dir, "test_config.yaml"))
    
    return test_config

if __name__ == "__main__":
    # When run directly, this can validate and display the current configuration
    try:
        print("Loading configuration...")
        config = load_config()
        print("Validating configuration...")
        validate_config(config)
        print("Configuration is valid.")
        
        # Display some key configuration values
        print("\nCurrent configuration:")
        print(f"- Sample rate: {config['data']['sample_rate']} Hz")
        print(f"- Using GPU: {config['system']['use_gpu']}")
        print(f"- Model type: {config['model']['type']}")
        print(f"- Batch size: {config['testing']['batch_size']}")
        print(f"- Metrics: {', '.join(config['testing']['metrics'])}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")