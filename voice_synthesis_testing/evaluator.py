# voice_synthesis_testing/evaluator.py
import numpy as np
import torch
import librosa
import jiwer
import scipy
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
from pathlib import Path

class Evaluator:
    """
    Implements various metrics for evaluating voice synthesis quality.
    """
    
    def __init__(self, config):
        """
        Initialize the evaluator with configuration
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize speech recognition model if needed for WER calculations
        self.asr_model = None
        if getattr(self.config, 'use_asr_for_wer', False):
            self._init_asr_model()
    
    def _init_asr_model(self):
        """Initialize automatic speech recognition model"""
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            model_name = getattr(self.config, 'asr_model', 'facebook/wav2vec2-base-960h')
            self.logger.info(f"Loading ASR model: {model_name}")
            
            self.asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.asr_model = self.asr_model.cuda()
                
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {str(e)}")
            self.asr_model = None
    
    def calculate_mcd(self, ref_audio: np.ndarray, 
                     synth_audio: np.ndarray, 
                     sr: int = 22050) -> float:
        """
        Calculate Mel Cepstral Distortion (MCD) between reference and synthesized audio
        
        Args:
            ref_audio: Reference audio
            synth_audio: Synthesized audio
            sr: Sample rate
            
        Returns:
            MCD score (lower is better)
        """
        # Extract MFCCs for both audio samples
        ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
        synth_mfcc = librosa.feature.mfcc(y=synth_audio, sr=sr, n_mfcc=13)
        
        # Dynamic time warping to align the sequences
        _, wp = librosa.sequence.dtw(ref_mfcc, synth_mfcc, backtrack=True)
        ref_indices = wp[:, 0]
        synth_indices = wp[:, 1]
        
        # Calculate Euclidean distance between aligned frames
        aligned_ref_mfcc = ref_mfcc[:, ref_indices]
        aligned_synth_mfcc = synth_mfcc[:, synth_indices]
        
        # Skip the first coefficient (energy)
        distances = np.sqrt(np.sum((aligned_ref_mfcc[1:] - aligned_synth_mfcc[1:]) ** 2, axis=0))
        
        # Calculate MCD
        mcd = np.mean(distances)
        
        return mcd
    
    def calculate_wer(self, reference_text: str, 
                     audio_path: str) -> float:
        """
        Calculate Word Error Rate using ASR
        
        Args:
            reference_text: Reference text
            audio_path: Path to synthesized audio
            
        Returns:
            WER score (lower is better)
        """
        if self.asr_model is None:
            self.logger.warning("ASR model not available for WER calculation")
            return -1.0
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)  # ASR models typically use 16kHz
            
            # Transcribe with ASR
            inputs = self.asr_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
                
            with torch.no_grad():
                logits = self.asr_model(inputs.input_values).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.batch_decode(predicted_ids)[0]
            
            # Calculate WER
            wer = jiwer.wer(reference_text, transcription)
            
            return wer
            
        except Exception as e:
            self.logger.error(f"Error calculating WER: {str(e)}")
            return -1.0
    
    def calculate_f0_rmse(self, ref_audio: np.ndarray, 
                         synth_audio: np.ndarray, 
                         sr: int = 22050) -> float:
        """
        Calculate RMSE of fundamental frequency between reference and synthesized audio
        
        Args:
            ref_audio: Reference audio
            synth_audio: Synthesized audio
            sr: Sample rate
            
        Returns:
            F0 RMSE (lower is better)
        """
        # Extract F0 for both audio samples
        ref_f0, voiced_flag, _ = librosa.pyin(
            ref_audio,
            fmin=65,
            fmax=800,
            sr=sr
        )
        
        synth_f0, synth_voiced_flag, _ = librosa.pyin(
            synth_audio,
            fmin=65,
            fmax=800,
            sr=sr
        )
        
        # Use DTW to align the sequences
        ref_f0_frames = ref_f0[voiced_flag]
        synth_f0_frames = synth_f0[synth_voiced_flag]
        
        if len(ref_f0_frames) == 0 or len(synth_f0_frames) == 0:
            return -1.0  # Cannot calculate
        
        # Convert to frames for DTW
        ref_f0_2d = ref_f0_frames.reshape(-1, 1)
        synth_f0_2d = synth_f0_frames.reshape(-1, 1)
        
        _, wp = librosa.sequence.dtw(ref_f0_2d.T, synth_f0_2d.T, backtrack=True)
        
        # Extract aligned frames
        ref_indices = wp[:, 0]
        synth_indices = wp[:, 1]
        
        aligned_ref_f0 = ref_f0_frames[ref_indices]
        aligned_synth_f0 = synth_f0_frames[synth_indices]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((aligned_ref_f0 - aligned_synth_f0) ** 2))
        
        return rmse
    
    def evaluate_audio_pair(self, reference_path: str, 
                           synthesized_path: str, 
                           reference_text: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a pair of reference and synthesized audio files
        
        Args:
            reference_path: Path to reference audio
            synthesized_path: Path to synthesized audio
            reference_text: Reference text (for WER calculation)
            
        Returns:
            Dictionary of metrics
        """
        # Load audio files
        ref_audio, ref_sr = librosa.load(reference_path, sr=None)
        synth_audio, synth_sr = librosa.load(synthesized_path, sr=None)
        
        # Resample if needed
        if ref_sr != synth_sr:
            self.logger.info(f"Resampling from {synth_sr} to {ref_sr}")
            synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=ref_sr)
            synth_sr = ref_sr
        
        # Calculate metrics
        metrics = {}
        
        # MCD
        metrics['mcd'] = self.calculate_mcd(ref_audio, synth_audio, ref_sr)
        
        # F0 RMSE
        metrics['f0_rmse'] = self.calculate_f0_rmse(ref_audio, synth_audio, ref_sr)
        
        # WER (if reference text is provided and ASR model is available)
        if reference_text and self.asr_model:
            metrics['wer'] = self.calculate_wer(reference_text, synthesized_path)
            
        return metrics
    
    def batch_evaluate(self, reference_dir: str, 
                      synthesized_dir: str, 
                      reference_texts: Optional[Dict[str, str]] = None) -> Dict[str, List[float]]:
        """
        Batch evaluate multiple audio pairs
        
        Args:
            reference_dir: Directory containing reference audio files
            synthesized_dir: Directory containing synthesized audio files
            reference_texts: Dictionary mapping filenames to reference texts
            
        Returns:
            Dictionary of lists of metrics
        """
        reference_files = sorted(os.listdir(reference_dir))
        synthesized_files = sorted(os.listdir(synthesized_dir))
        
        # Ensure matching files
        common_files = set(reference_files).intersection(set(synthesized_files))
        self.logger.info(f"Found {len(common_files)} matching files for evaluation")
        
        # Initialize results
        results = {
            'mcd': [],
            'f0_rmse': []
        }
        
        if reference_texts and self.asr_model:
            results['wer'] = []
        
        # Evaluate each pair
        for filename in common_files:
            ref_path = os.path.join(reference_dir, filename)
            synth_path = os.path.join(synthesized_dir, filename)
            
            # Get reference text if available
            ref_text = None
            if reference_texts and filename in reference_texts:
                ref_text = reference_texts[filename]
                
            # Evaluate
            metrics = self.evaluate_audio_pair(ref_path, synth_path, ref_text)
            
            # Store results
            for metric_name, metric_value in metrics.items():
                results[metric_name].append(metric_value)
        
        # Calculate aggregated statistics
        aggregated = {}
        for metric_name, values in results.items():
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)
        
        # Combine individual and aggregated results
        final_results = {**results, **aggregated}
        
        return final_results
    
    def generate_report(self, metrics: Dict[str, Union[float, List[float]]], 
                       output_path: str):
        """
        Generate a report from evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
            output_path: Path to save the report
        """
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"Saved evaluation report to {output_path}")