# voice_synthesis_testing/feature_processor.py
import numpy as np
import librosa
import torch
from typing import Dict, List, Optional, Union, Tuple
import logging

class FeatureProcessor:
    """
    Extracts and manipulates audio features (MFCCs, spectrograms, etc.)
    """
    
    def __init__(self, config):
        """
        Initialize the feature processor with configuration
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set default parameters from config or use standard values
        self.sample_rate = config.data.get('sample_rate', 22050)
        self.n_fft = config.data.get('n_fft', 1024)
        self.hop_length = config.data.get('hop_length', 256)
        self.win_length = config.data.get('win_length', 1024)
        self.n_mels = config.data.get('n_mels', 80)
        self.n_mfcc = config.data.get('n_mfcc', 13)
        self.fmin = config.data.get('fmin', 0)
        self.fmax = config.data.get('fmax', 8000)
    
    def load_audio(self, file_path: str, 
                  target_sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio file and convert to the desired sample rate
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (uses config default if None)
            
        Returns:
            Audio as numpy array
        """
        if target_sr is None:
            target_sr = self.sample_rate
            
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio
        except Exception as e:
            self.logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def extract_melspectrogram(self, audio: np.ndarray, 
                               sr: Optional[int] = None) -> np.ndarray:
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Audio signal
            sr: Sample rate (uses config default if None)
            
        Returns:
            Mel spectrogram
        """
        if sr is None:
            sr = self.sample_rate
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_mfcc(self, audio: np.ndarray, 
                    sr: Optional[int] = None) -> np.ndarray:
        """
        Extract MFCCs from audio
        
        Args:
            audio: Audio signal
            sr: Sample rate (uses config default if None)
            
        Returns:
            MFCCs
        """
        if sr is None:
            sr = self.sample_rate
            
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Concatenate features
        mfcc_features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        return mfcc_features
    
    def extract_f0(self, audio: np.ndarray, 
                  sr: Optional[int] = None) -> np.ndarray:
        """
        Extract fundamental frequency (F0) contour
        
        Args:
            audio: Audio signal
            sr: Sample rate (uses config default if None)
            
        Returns:
            F0 contour
        """
        if sr is None:
            sr = self.sample_rate
            
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.config.f0_min if hasattr(self.config, 'f0_min') else 65,
            fmax=self.config.f0_max if hasattr(self.config, 'f0_max') else 800,
            sr=sr
        )
        
        return f0
    
    def extract_features(self, audio: np.ndarray, 
                        sr: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio
        
        Args:
            audio: Audio signal
            sr: Sample rate (uses config default if None)
            
        Returns:
            Dictionary of features
        """
        if sr is None:
            sr = self.sample_rate
            
        # Extract different features
        features = {
            'melspectrogram': self.extract_melspectrogram(audio, sr),
            'mfcc': self.extract_mfcc(audio, sr)
        }
        
        # Add F0 if configured
        if getattr(self.config, 'extract_f0', False):
            features['f0'] = self.extract_f0(audio, sr)
            
        return features
    
    def normalize_features(self, features: Dict[str, np.ndarray], 
                          stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Normalize features using provided statistics
        
        Args:
            features: Dictionary of features
            stats: Dictionary of statistics (mean, std) for each feature type
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        
        for feat_name, feat_data in features.items():
            if feat_name in stats:
                mean = stats[feat_name]['mean']
                std = stats[feat_name]['std']
                
                # Handle dimensionality
                if feat_data.ndim != mean.ndim:
                    # Reshape statistics to match feature dimensions
                    if feat_data.ndim > mean.ndim:
                        for _ in range(feat_data.ndim - mean.ndim):
                            mean = mean[..., np.newaxis]
                            std = std[..., np.newaxis]
                    else:
                        for _ in range(mean.ndim - feat_data.ndim):
                            feat_data = feat_data[..., np.newaxis]
                
                # Normalize
                normalized[feat_name] = (feat_data - mean) / (std + 1e-8)
            else:
                # Pass through unchanged if no stats available
                normalized[feat_name] = feat_data
                
        return normalized
    
    def preprocess_audio_file(self, file_path: str, 
                             normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Process an audio file to extract features
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize features
            
        Returns:
            Dictionary of features
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Extract features
        features = self.extract_features(audio, self.sample_rate)
        
        # Normalize if requested
        if normalize and hasattr(self.config, 'feature_stats'):
            features = self.normalize_features(features, self.config.feature_stats)
            
        return features
    
    def batch_process_files(self, file_paths: List[str], 
                           normalize: bool = True) -> Dict[str, List[np.ndarray]]:
        """
        Process multiple audio files
        
        Args:
            file_paths: List of paths to audio files
            normalize: Whether to normalize features
            
        Returns:
            Dictionary of lists of features
        """
        results = {
            'melspectrogram': [],
            'mfcc': []
        }
        
        if getattr(self.config, 'extract_f0', False):
            results['f0'] = []
        
        for file_path in file_paths:
            features = self.preprocess_audio_file(file_path, normalize)
            
            for feat_name, feat_data in features.items():
                results[feat_name].append(feat_data)
                
        return results