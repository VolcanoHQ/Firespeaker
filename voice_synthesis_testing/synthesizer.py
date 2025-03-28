# voice_synthesis_testing/synthesizer.py
import os
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

class Synthesizer:
    """
    Main engine for text-to-speech conversion.
    Provides a unified interface for different voice synthesis models.
    """
    
    def __init__(self, config, device=None):
        """
        Initialize synthesizer with configuration.
        
        Args:
            config: Configuration object or dictionary
            device: Torch device (will use CUDA if available by default)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device (use GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Dictionary to store loaded models
        self.models = {}
        
        # Load default model if specified in config
        if hasattr(config, 'default_model') and config.default_model:
            self.load_model(config.default_model)
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a voice synthesis model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = Path(self.config.model_dir) / model_name
            
            if not model_path.exists():
                self.logger.error(f"Model {model_name} not found at {model_path}")
                return False
            
            # Import appropriate model handler based on model type
            # This is a simplified example - in practice, you would have more logic here
            # to handle different model architectures (VITS, FastSpeech2, etc.)
            if model_name.startswith("vits_"):
                from .models.vits_handler import VITSModel
                self.models[model_name] = VITSModel(model_path, self.device)
            elif model_name.startswith("fastspeech_"):
                from .models.fastspeech_handler import FastSpeechModel
                self.models[model_name] = FastSpeechModel(model_path, self.device)
            else:
                # Generic model loader
                from .models.generic_handler import GenericModel
                self.models[model_name] = GenericModel(model_path, self.device)
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def synthesize(self, 
                  text: str, 
                  model_name: Optional[str] = None,
                  speaker_id: Optional[int] = None,
                  speed: float = 1.0,
                  pitch_adjustment: float = 0.0,
                  energy_adjustment: float = 0.0,
                  save_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using the specified model.
        
        Args:
            text: Input text to synthesize
            model_name: Name of the model to use (uses default if None)
            speaker_id: Speaker ID for multi-speaker models
            speed: Speech speed factor (1.0 is normal)
            pitch_adjustment: Pitch adjustment in semitones
            energy_adjustment: Energy/volume adjustment
            save_path: Path to save the audio file (optional)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Use default model if none specified
        if model_name is None:
            if hasattr(self.config, 'default_model'):
                model_name = self.config.default_model
            else:
                raise ValueError("No model specified and no default model in config")
        
        # Load model if not already loaded
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                raise ValueError(f"Failed to load model: {model_name}")
        
        # Prepare synthesis parameters
        params = {
            "text": text,
            "speaker_id": speaker_id,
            "speed": speed,
            "pitch_adjustment": pitch_adjustment,
            "energy_adjustment": energy_adjustment
        }
        
        # Perform synthesis
        self.logger.info(f"Synthesizing with model {model_name}: '{text[:50]}...'")
        audio, sample_rate = self.models[model_name].synthesize(**params)
        
        # Save audio if requested
        if save_path:
            self._save_audio(audio, sample_rate, save_path)
            
        return audio, sample_rate
    
    def _save_audio(self, audio: np.ndarray, sample_rate: int, path: str):
        """Helper method to save audio to file"""
        import soundfile as sf
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, sample_rate)
        self.logger.info(f"Saved audio to {path}")
    
    def batch_synthesize(self, 
                        texts: List[str], 
                        output_dir: str,
                        model_name: Optional[str] = None,
                        filename_prefix: str = "synth_",
                        **kwargs) -> List[str]:
        """
        Synthesize multiple texts and save to files.
        
        Args:
            texts: List of input texts
            output_dir: Directory to save output files
            model_name: Model to use
            filename_prefix: Prefix for output filenames
            **kwargs: Additional parameters for synthesis
            
        Returns:
            List of paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i, text in enumerate(texts):
            filename = f"{filename_prefix}{i:04d}.wav"
            output_path = os.path.join(output_dir, filename)
            
            self.synthesize(
                text=text,
                model_name=model_name,
                save_path=output_path,
                **kwargs
            )
            
            output_files.append(output_path)
        
        return output_files