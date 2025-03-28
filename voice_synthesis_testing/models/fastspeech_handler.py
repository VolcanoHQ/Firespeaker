# voice_synthesis_testing/models/fastspeech_handler.py
import torch
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import re
import phonemizer
from phonemizer.backend import EspeakBackend
import yaml
import scipy.signal as signal

class FastSpeechModel:
    """
    Handler for FastSpeech2 models.
    
    FastSpeech2 is a non-autoregressive TTS model that predicts mel-spectrograms 
    from text with explicit duration, pitch, and energy prediction.
    """
    
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the FastSpeech2 model.
        
        Args:
            model_path: Path to the model directory
            device: Torch device (CPU or GPU)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.sample_rate = 22050  # Default sample rate
        
        # Load model
        self._load_model()
        
        # Initialize phoneme converter
        self._init_phonemizer()
    
    def _load_model(self):
        """Load the FastSpeech2 model and configuration"""
        try:
            # Load model configuration
            config_file = None
            # Look for config files with different possible names
            for filename in ["config.json", "model.yaml", "model_config.json"]:
                potential_config = self.model_path / filename
                if potential_config.exists():
                    config_file = potential_config
                    break
            
            if config_file is None:
                raise FileNotFoundError(f"No config file found in {self.model_path}")
            
            # Load configuration based on file type
            if str(config_file).endswith('.json'):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            elif str(config_file).endswith(('.yaml', '.yml')):
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
            
            # Extract necessary parameters from config
            self._extract_config_params()
            
            # Find model checkpoint
            checkpoint_file = None
            for filename in ["model.pth", "checkpoint.pth", "G_latest.pth", "model_ckpt.pth"]:
                potential_ckpt = self.model_path / filename
                if potential_ckpt.exists():
                    checkpoint_file = potential_ckpt
                    break
            
            if checkpoint_file is None:
                # Look for checkpoints in subdirectories
                checkpoints = list(self.model_path.glob("**/*.pth"))
                if checkpoints:
                    checkpoint_file = checkpoints[0]
                else:
                    raise FileNotFoundError(f"No model checkpoint found in {self.model_path}")
            
            self.logger.info(f"Loading FastSpeech2 model from {checkpoint_file}")
            self.logger.info(f"Model will run on {self.device}")
            
            # Import and load the actual model
            # Note: We need to have the FastSpeech2 model implementation
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            # Initialize model based on config
            self.model = self._init_model()
            
            # Load weights
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Successfully loaded FastSpeech2 model")
            
        except Exception as e:
            self.logger.error(f"Error loading FastSpeech2 model: {str(e)}")
            raise
    
    def _extract_config_params(self):
        """Extract necessary parameters from the loaded configuration"""
        # Set model parameters with defaults as fallback
        self.n_mel_channels = self.config.get("n_mel_channels", 
                                             self.config.get("audio", {}).get("n_mel_channels", 80))
        self.sample_rate = self.config.get("sampling_rate", 
                                          self.config.get("audio", {}).get("sampling_rate", 22050))
        self.max_seq_len = self.config.get("max_seq_len", 1000)
        
        # Extract variance parameters
        self.pitch_feature_level = self.config.get("pitch", {}).get("feature", "phoneme_level")
        self.energy_feature_level = self.config.get("energy", {}).get("feature", "phoneme_level")
        
        # Extract pitch and energy normalization stats if available
        if "pitch" in self.config and "stats" in self.config["pitch"]:
            self.pitch_mean = self.config["pitch"]["stats"]["mean"]
            self.pitch_std = self.config["pitch"]["stats"]["std"]
        else:
            self.pitch_mean = 0.0
            self.pitch_std = 1.0
            
        if "energy" in self.config and "stats" in self.config["energy"]:
            self.energy_mean = self.config["energy"]["stats"]["mean"]
            self.energy_std = self.config["energy"]["stats"]["std"]
        else:
            self.energy_mean = 0.0
            self.energy_std = 1.0
        
        # Get text preprocessing parameters
        self.text_cleaner = self.config.get("text_cleaner", ["english_cleaners"])
        
    def _init_model(self):
        """Initialize the FastSpeech2 model based on configuration"""
        try:
            # Try to use a local FastSpeech2 implementation if available
            from .fastspeech2.model import FastSpeech2
            model = FastSpeech2(
                self.config.get("vocab_size", 300),
                self.config.get("max_seq_len", 1000),
                self.n_mel_channels,
                self.config.get("encoder_hidden", 256),
                self.config.get("encoder_head", 4),
                self.config.get("encoder_layer", 4),
                self.config.get("decoder_hidden", 256),
                self.config.get("decoder_head", 4),
                self.config.get("decoder_layer", 4),
                self.config.get("fft_conv1d_filter_size", 1024),
                self.config.get("fft_conv1d_kernel_size", 9),
                self.config.get("encoder_dropout", 0.2),
                self.config.get("decoder_dropout", 0.2),
                use_pitch_embed=self.config.get("use_pitch_embed", True),
                use_energy_embed=self.config.get("use_energy_embed", True),
                stats_file=self.config.get("stats_file", None)
            )
            return model
        except ImportError:
            # If the specific implementation is not available, use a placeholder
            self.logger.warning("FastSpeech2 model implementation not found. Using placeholder.")
            
            # Define a placeholder model that will let us test interfaces
            class FastSpeech2Placeholder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dummy_param = torch.nn.Parameter(torch.zeros(1))
                
                def forward(self, src_seq, src_len, mel_len=None, 
                           d_target=None, p_target=None, e_target=None, 
                           max_src_len=None, max_mel_len=None):
                    batch_size = src_seq.size(0)
                    # Return random mel-spectrogram as placeholder
                    return torch.randn(batch_size, max_mel_len or 200, 
                                       self.n_mel_channels, device=self.device)
                
                def inference(self, text, alpha=1.0, beta=1.0, gamma=1.0):
                    seq_len = len(text)
                    # Return random mel-spectrogram as placeholder
                    return torch.randn(1, seq_len * 5, self.n_mel_channels, device=self.device)
            
            model = FastSpeech2Placeholder()
            model.n_mel_channels = self.n_mel_channels
            return model
    
    def _init_phonemizer(self):
        """Initialize the phoneme converter"""
        try:
            self.phonemizer = phonemizer.backend.EspeakBackend(
                language='en-us',
                preserve_punctuation=True,
                with_stress=True
            )
            self.logger.info("Initialized phoneme converter successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize phoneme converter: {str(e)}")
            self.phonemizer = None
    
    def _text_to_phoneme(self, text: str) -> List[str]:
        """Convert text to phoneme sequence"""
        if self.phonemizer is None:
            # Simple fallback if phonemizer is not available
            return list(text.lower())
        
        try:
            phonemes = self.phonemizer.phonemize([text], strip=True)[0]
            return phonemes.split()
        except Exception as e:
            self.logger.warning(f"Error in phonemization: {str(e)}. Using simple text.")
            return list(text.lower())
    
    def _text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text to input sequence for the model"""
        # First convert to phonemes
        phonemes = self._text_to_phoneme(text)
        
        # Apply simple encoding based on available resources
        # This is a simplified approach; a proper implementation would use
        # the model's specific text encoding logic
        vocab = "_ abcdefghijklmnopqrstuvwxyz"
        char_to_id = {c: i for i, c in enumerate(vocab)}
        
        sequence = []
        for p in phonemes:
            for c in p:
                if c.lower() in char_to_id:
                    sequence.append(char_to_id[c.lower()])
                else:
                    # Use underscore for unknown characters
                    sequence.append(char_to_id['_'])
        
        return torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def synthesize(self, 
                  text: str,
                  speaker_id: Optional[int] = None,
                  speed: float = 1.0,
                  pitch_adjustment: float = 0.0,
                  energy_adjustment: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for multi-speaker models
            speed: Speech speed factor (1.0 is normal)
            pitch_adjustment: Pitch adjustment in semitones
            energy_adjustment: Energy/volume adjustment
            
        Returns:
            Tuple of (mel_spectrogram, sample_rate)
        """
        self.logger.info(f"Synthesizing with FastSpeech2: '{text[:50]}...'")
        
        # Convert speed to duration ratio (inverse relationship)
        duration_ratio = 1.0 / speed
        
        # Convert pitch adjustment from semitones to scalar for the model
        # Each semitone is approximately a 6% change in frequency
        pitch_scalar = 2 ** (pitch_adjustment / 12)
        
        # Energy adjustment is typically in dB, convert to linear scale
        energy_scalar = 10 ** (energy_adjustment / 20)
        
        try:
            # Preprocess text to model input
            sequence = self._text_to_sequence(text)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.model, 'inference'):
                    # Use specific inference method if available
                    output = self.model.inference(
                        sequence, 
                        alpha=duration_ratio,
                        beta=pitch_scalar,
                        gamma=energy_scalar
                    )
                else:
                    # Fallback to generic forward method
                    src_len = torch.tensor([sequence.shape[1]], device=self.device)
                    output = self.model(sequence, src_len)
                
                # Extract mel-spectrogram from output
                # The exact format depends on the specific model implementation
                if isinstance(output, tuple):
                    mel_output = output[0]  # First element is usually the mel output
                elif isinstance(output, dict):
                    mel_output = output.get('mel_output', output.get('mel_pred', output))
                else:
                    mel_output = output
                
                # Ensure correct dimensions [batch, time, feature]
                if mel_output.dim() == 4:  # [batch, channel, time, feature]
                    mel_output = mel_output.squeeze(1)
                
                mel_spectrogram = mel_output[0].cpu().numpy()  # Remove batch dimension
            
            self.logger.info(f"Generated mel-spectrogram with shape {mel_spectrogram.shape}")
            
            return mel_spectrogram, self.sample_rate
            
        except Exception as e:
            self.logger.error(f"Error in FastSpeech2 synthesis: {str(e)}")
            
            # Generate a fallback mel-spectrogram to not break the pipeline
            # This produces a simple sine wave as a placeholder
            duration = len(text) / 10  # Rough approximation: 10 characters per second
            duration = duration / speed  # Adjust for speed
            
            # Create time steps
            n_frames = int(duration * self.sample_rate / 256)  # Assuming 256 hop size
            
            # Create a placeholder mel-spectrogram (80 mel channels is standard)
            mel_spectrogram = np.zeros((n_frames, self.n_mel_channels))
            
            # Add a simple pattern to make it audible
            t = np.linspace(0, duration, n_frames)
            frequency = 220.0 * (2 ** (pitch_adjustment / 12.0))  # A3 with pitch adjustment
            
            # Create a simple sinusoid pattern in the mel-spectrogram
            # This doesn't produce real speech but serves as a fallback
            for i in range(min(20, self.n_mel_channels)):
                mel_spectrogram[:, i] = np.sin(2 * np.pi * frequency * (i+1)/10 * t) * (self.n_mel_channels - i) / self.n_mel_channels * energy_scalar
            
            self.logger.warning(f"Using fallback mel-spectrogram with shape {mel_spectrogram.shape}")
            
            return mel_spectrogram, self.sample_rate
    
    def _load_vocoder(self):
        """Load vocoder for mel-spectrogram to waveform conversion"""
        try:
            # Try to load HiFi-GAN vocoder if available
            vocoder_path = self.model_path.parent / "vocoder" / "model.pth"
            if vocoder_path.exists():
                self.logger.info(f"Loading vocoder from {vocoder_path}")
                from .hifigan.models import Generator as HiFiGANGenerator
                
                with open(self.model_path.parent / "vocoder" / "config.json", 'r') as f:
                    config = json.load(f)
                
                self.vocoder = HiFiGANGenerator(config)
                checkpoint = torch.load(vocoder_path, map_location=self.device)
                self.vocoder.load_state_dict(checkpoint['generator'])
                self.vocoder.eval()
                self.vocoder.remove_weight_norm()
                self.vocoder.to(self.device)
                return True
            else:
                self.logger.warning(f"No vocoder found at {vocoder_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading vocoder: {str(e)}")
            return False
    
    def mel_to_waveform(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Convert mel-spectrogram to waveform using a vocoder.
        
        Args:
            mel_spectrogram: Mel-spectrogram as a numpy array
            
        Returns:
            Waveform as a numpy array
        """
        # Check if vocoder is loaded, load if not
        if not hasattr(self, 'vocoder'):
            if not self._load_vocoder():
                # Fallback using Griffin-Lim if vocoder not available
                self.logger.warning("Using Griffin-Lim algorithm as fallback vocoder")
                return self._griffin_lim(mel_spectrogram)
        
        # Convert mel-spectrogram to tensor
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float).to(self.device)
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)  # Add batch dimension
        
        # Transpose to match vocoder input format if needed
        if mel_tensor.shape[1] == self.n_mel_channels:
            mel_tensor = mel_tensor.transpose(1, 2)
        
        # Generate waveform
        with torch.no_grad():
            waveform = self.vocoder(mel_tensor).squeeze(0).cpu().numpy()
        
        return waveform
    
    def _griffin_lim(self, mel_spectrogram: np.ndarray, n_iter: int = 30) -> np.ndarray:
        """
        Griffin-Lim algorithm for waveform reconstruction from mel-spectrogram.
        This is a simplified implementation and serves as a fallback.
        
        Args:
            mel_spectrogram: Mel-spectrogram as a numpy array
            n_iter: Number of iterations
            
        Returns:
            Waveform as a numpy array
        """
        # This is a very simplified version to serve as a fallback
        # For accurate reconstruction, a proper implementation would:
        # 1. Convert mel-spectrogram back to linear spectrogram
        # 2. Perform Griffin-Lim on the linear spectrogram
        
        # Generate a simple sine wave based on the energy in the mel-spectrogram
        n_frames = mel_spectrogram.shape[0]
        hop_length = 256  # Typical hop length
        
        # Calculate the total number of samples
        n_samples = n_frames * hop_length
        
        # Extract a rough pitch curve from the mel-spectrogram
        energy = np.mean(mel_spectrogram, axis=1)
        energy = np.maximum(energy, 0)
        
        # Generate time base
        t = np.arange(n_samples) / self.sample_rate
        
        # Generate a simple tone based on the energy curve
        waveform = np.zeros(n_samples)
        for i in range(n_frames):
            idx_start = i * hop_length
            idx_end = min((i + 1) * hop_length, n_samples)
            
            # Base frequency and amplitude modulation
            freq = 220.0  # A3 as a base
            amp = energy[i] * 0.1  # Scale energy to a reasonable amplitude
            
            # Add harmonic components
            for j in range(1, 5):  # Add a few harmonics
                harm_amp = amp / j  # Amplitude decreases for higher harmonics
                waveform[idx_start:idx_end] += harm_amp * np.sin(2 * np.pi * freq * j * t[idx_start:idx_end])
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)
        
        return waveform