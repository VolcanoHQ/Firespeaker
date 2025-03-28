# voice_synthesis_testing/models/bark_handler.py
import numpy as np
import torch
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import nltk

class BarkModel:
    """
    Handler for Bark text-to-speech model.
    
    Bark is a transformer-based text-to-audio model that can generate
    realistic speech, music, and sound effects.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        """
        Initialize the Bark model.
        
        Args:
            model_path: Path to the model directory (optional for Bark)
            device: Torch device (CPU or GPU)
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device (use GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Set environment variable for device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0 if self.device.type == "cuda" else "")
        
        # Initialize NLTK for sentence splitting
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)
        
        self.sample_rate = 24000  # Bark's default sample rate
        
        # Load Bark
        self._load_model()
    
    def _load_model(self):
        """Load the Bark model"""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from bark.generation import generate_text_semantic, semantic_to_waveform
            
            # Store references to Bark functions
            self.SAMPLE_RATE = SAMPLE_RATE
            self.generate_audio = generate_audio
            self.generate_text_semantic = generate_text_semantic
            self.semantic_to_waveform = semantic_to_waveform
            
            # Preload models
            self.logger.info("Preloading Bark models")
            preload_models()
            
            self.logger.info("Successfully loaded Bark model")
            
        except ImportError:
            self.logger.error("Bark not installed. Please install with 'pip install bark'")
            raise ImportError("Bark not installed. Please install with 'pip install bark'")
    
    def synthesize(self, 
                  text: str,
                  speaker_id: Optional[str] = "v2/en_speaker_6",
                  speed: float = 1.0,
                  pitch_adjustment: float = 0.0,
                  energy_adjustment: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Bark.
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID (Bark prompt)
            speed: Speech speed factor (approximated in Bark)
            pitch_adjustment: Not directly supported in Bark
            energy_adjustment: Not directly supported in Bark
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self.logger.info(f"Synthesizing with Bark: '{text[:50]}...'")
        
        # Bark doesn't directly support speed/pitch/energy adjustments
        # Speed can be approximated by adjusting pause durations in post-processing
        if speed != 1.0 or pitch_adjustment != 0.0 or energy_adjustment != 0.0:
            self.logger.warning("Bark doesn't directly support speed, pitch, or energy adjustments")
        
        try:
            # For longer texts, split into sentences
            if len(text) > 200:
                return self._synthesize_long_text(text, speaker_id)
            
            # For short texts, generate directly
            audio_array = self.generate_audio(text, history_prompt=speaker_id)
            
            self.logger.info(f"Generated {len(audio_array)/self.SAMPLE_RATE:.2f}s of audio")
            
            return audio_array, self.SAMPLE_RATE
            
        except Exception as e:
            self.logger.error(f"Error in Bark synthesis: {str(e)}")
            self.logger.error("Returning empty audio")
            
            # Return 1 second of silence as fallback
            return np.zeros(self.SAMPLE_RATE), self.SAMPLE_RATE
    
    def _synthesize_long_text(self, 
                             text: str, 
                             speaker_id: str, 
                             min_eos_p: float = 0.05) -> Tuple[np.ndarray, int]:
        """
        Synthesize longer text by splitting into sentences.
        
        Args:
            text: Long text to synthesize
            speaker_id: Speaker ID (Bark prompt)
            min_eos_p: Controls how likely generation is to end
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        sentences = nltk.sent_tokenize(text)
        self.logger.info(f"Split text into {len(sentences)} sentences")
        
        silence = np.zeros(int(0.25 * self.SAMPLE_RATE))  # quarter second of silence
        pieces = []
        
        for i, sentence in enumerate(sentences):
            self.logger.info(f"Generating sentence {i+1}/{len(sentences)}")
            
            try:
                # Generate semantic tokens with controlled ending probability
                semantic_tokens = self.generate_text_semantic(
                    sentence,
                    history_prompt=speaker_id,
                    temp=0.6,
                    min_eos_p=min_eos_p
                )
                
                # Convert semantic tokens to audio
                audio_array = self.semantic_to_waveform(semantic_tokens, history_prompt=speaker_id)
                
                # Add to pieces with silence
                pieces += [audio_array, silence.copy()]
                
            except Exception as e:
                self.logger.error(f"Error generating sentence {i+1}: {str(e)}")
                # Continue with next sentence
        
        if not pieces:
            self.logger.error("No audio generated for any sentence")
            return np.zeros(self.SAMPLE_RATE), self.SAMPLE_RATE
        
        # Concatenate all pieces
        return np.concatenate(pieces), self.SAMPLE_RATE
    
    def synthesize_dialogue(self, 
                          script: List[str], 
                          speaker_lookup: Dict[str, str]) -> Tuple[np.ndarray, int]:
        """
        Synthesize a dialogue script with multiple speakers.
        
        Args:
            script: List of dialogue lines in format "Speaker: Text"
            speaker_lookup: Dictionary mapping speaker names to speaker IDs
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self.logger.info(f"Synthesizing dialogue with {len(script)} turns")
        
        pieces = []
        silence = np.zeros(int(0.5 * self.SAMPLE_RATE))  # half second of silence
        
        for i, line in enumerate(script):
            try:
                # Split the line into speaker and text
                if ": " not in line:
                    self.logger.warning(f"Invalid dialogue line format: {line}")
                    continue
                    
                speaker, text = line.split(": ", 1)
                
                if speaker not in speaker_lookup:
                    self.logger.warning(f"Speaker '{speaker}' not found in speaker_lookup")
                    continue
                
                self.logger.info(f"Generating line {i+1}/{len(script)} for {speaker}")
                
                # Generate audio for this line
                audio_array = self.generate_audio(text, history_prompt=speaker_lookup[speaker])
                
                # Add to pieces with silence
                pieces += [audio_array, silence.copy()]
                
            except Exception as e:
                self.logger.error(f"Error generating dialogue line {i+1}: {str(e)}")
                # Continue with next line
        
        if not pieces:
            self.logger.error("No audio generated for any dialogue line")
            return np.zeros(self.SAMPLE_RATE), self.SAMPLE_RATE
        
        # Concatenate all pieces
        return np.concatenate(pieces), self.SAMPLE_RATE