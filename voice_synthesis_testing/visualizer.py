# voice_synthesis_testing/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from typing import Dict, List, Optional, Union, Tuple
import logging

class Visualizer:
    """
    Creates visualizations for audio data and evaluation results.
    """
    
    def __init__(self, config):
        """
        Initialize the visualizer with configuration
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set figure size and style
        self.figsize = config.data.get('figsize', (12, 8))
        plt.style.use(config.data.get('plot_style', 'default'))
    
    def plot_waveform(self, audio: np.ndarray, sr: int, 
                     title: str = "Waveform", 
                     save_path: Optional[str] = None):
        """
        Plot audio waveform
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved waveform plot to {save_path}")
        else:
            plt.show()
    
    def plot_spectrogram(self, audio: np.ndarray, sr: int, 
                        title: str = "Spectrogram", 
                        save_path: Optional[str] = None):
        """
        Plot spectrogram
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        plt.figure(figsize=self.figsize)
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved spectrogram plot to {save_path}")
        else:
            plt.show()
    
    def plot_melspectrogram(self, audio: np.ndarray, sr: int, 
                           title: str = "Mel Spectrogram", 
                           save_path: Optional[str] = None):
        """
        Plot mel spectrogram
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        plt.figure(figsize=self.figsize)
        
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved mel spectrogram plot to {save_path}")
        else:
            plt.show()
    
    def plot_f0_contour(self, audio: np.ndarray, sr: int, 
                       title: str = "F0 Contour", 
                       save_path: Optional[str] = None):
        """
        Plot fundamental frequency (F0) contour
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        plt.figure(figsize=self.figsize)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=65,
            fmax=800,
            sr=sr
        )
        
        times = librosa.times_like(f0, sr=sr)
        
        plt.plot(times, f0, label='F0', linewidth=2)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, 800)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved F0 contour plot to {save_path}")
        else:
            plt.show()
    
    def plot_comparison(self, ref_audio: np.ndarray, synth_audio: np.ndarray, 
                       sr: int, metrics: Optional[Dict[str, float]] = None, 
                       title: str = "Reference vs Synthesized", 
                       save_path: Optional[str] = None):
        """
        Plot comparison between reference and synthesized audio
        
        Args:
            ref_audio: Reference audio
            synth_audio: Synthesized audio
            sr: Sample rate
            metrics: Dictionary of metrics to display
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Waveforms
        axes[0, 0].plot(np.linspace(0, len(ref_audio)/sr, len(ref_audio)), ref_audio)
        axes[0, 0].set_title("Reference Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        
        axes[0, 1].plot(np.linspace(0, len(synth_audio)/sr, len(synth_audio)), synth_audio)
        axes[0, 1].set_title("Synthesized Waveform")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude")
        
        # Spectrograms
        D_ref = librosa.amplitude_to_db(np.abs(librosa.stft(ref_audio)), ref=np.max)
        librosa.display.specshow(D_ref, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 0])
        axes[1, 0].set_title("Reference Spectrogram")
        
        D_synth = librosa.amplitude_to_db(np.abs(librosa.stft(synth_audio)), ref=np.max)
        img = librosa.display.specshow(D_synth, sr=sr, x_axis='time', y_axis='log', ax=axes[1, 1])
        axes[1, 1].set_title("Synthesized Spectrogram")
        
        # Add colorbar
        fig.colorbar(img, ax=[axes[1, 0], axes[1, 1]], format='%+2.0f dB')
        
        # F0 contours
        f0_ref, voiced_flag_ref, _ = librosa.pyin(ref_audio, fmin=65, fmax=800, sr=sr)
        times_ref = librosa.times_like(f0_ref, sr=sr)
        axes[2, 0].plot(times_ref, f0_ref, label='F0', linewidth=2)
        axes[2, 0].set_title("Reference F0 Contour")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Frequency (Hz)")
        axes[2, 0].set_ylim(0, 800)
        
        f0_synth, voiced_flag_synth, _ = librosa.pyin(synth_audio, fmin=65, fmax=800, sr=sr)
        times_synth = librosa.times_like(f0_synth, sr=sr)
        axes[2, 1].plot(times_synth, f0_synth, label='F0', linewidth=2)
        axes[2, 1].set_title("Synthesized F0 Contour")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_ylabel("Frequency (Hz)")
        axes[2, 1].set_ylim(0, 800)
        
        # Add metrics if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            plt.figtext(0.5, 0.01, f"Metrics:\n{metrics_text}", 
                       ha="center", fontsize=12, 
                       bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
    
    def plot_metrics_summary(self, metrics: Dict[str, List[float]], 
                           title: str = "Evaluation Metrics Summary", 
                           save_path: Optional[str] = None):
        """
        Plot summary of evaluation metrics
        
        Args:
            metrics: Dictionary of metrics lists
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        # Filter out aggregated metrics (those with _mean, _std, etc. suffix)
        plot_metrics = {k: v for k, v in metrics.items() 
                       if isinstance(v, list) and not any(suffix in k for suffix in 
                                                         ['_mean', '_std', '_min', '_max'])}
        
        n_metrics = len(plot_metrics)
        if n_metrics == 0:
            self.logger.warning("No metrics to plot")
            return
            
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        fig.suptitle(title, fontsize=16)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
            
        for i, (metric_name, values) in enumerate(plot_metrics.items()):
            axes[i].boxplot(values)
            axes[i].set_title(f"{metric_name} Distribution")
            axes[i].set_ylabel(metric_name)
            
            # Add individual points
            x = np.random.normal(1, 0.04, size=len(values))
            axes[i].scatter(x, values, alpha=0.5)
            
            # Add mean line
            if f"{metric_name}_mean" in metrics:
                mean_val = metrics[f"{metric_name}_mean"]
                axes[i].axhline(mean_val, color='red', linestyle='--', 
                               label=f"Mean: {mean_val:.4f}")
                axes[i].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved metrics summary plot to {save_path}")
        else:
            plt.show()
    
    def plot_model_comparison(self, metrics_by_model: Dict[str, Dict[str, float]], 
                            metric_names: List[str], 
                            title: str = "Model Comparison", 
                            save_path: Optional[str] = None):
        """
        Plot comparison of metrics across different models
        
        Args:
            metrics_by_model: Dictionary mapping model names to their metrics
            metric_names: List of metric names to include in the plot
            title: Plot title
            save_path: Path to save the plot (if None, displays it)
        """
        n_models = len(metrics_by_model)
        n_metrics = len(metric_names)
        
        if n_models == 0 or n_metrics == 0:
            self.logger.warning("No models or metrics to plot")
            return
            
        # Set up the figure
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        fig.suptitle(title, fontsize=16)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
            
        model_names = list(metrics_by_model.keys())
        
        for i, metric_name in enumerate(metric_names):
            # Extract values for this metric across all models
            values = [metrics_by_model[model].get(f"{metric_name}_mean", 0) 
                     for model in model_names]
            
            # Create bar chart
            bar_positions = np.arange(n_models)
            axes[i].bar(bar_positions, values)
            
            # Add error bars if std deviation is available
            errors = [metrics_by_model[model].get(f"{metric_name}_std", 0) 
                     for model in model_names]
            axes[i].errorbar(bar_positions, values, yerr=errors, fmt='none', 
                            ecolor='black', capsize=5)
            
            # Labels and formatting
            axes[i].set_title(metric_name)
            axes[i].set_xticks(bar_positions)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].set_ylabel(metric_name)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved model comparison plot to {save_path}")
        else:
            plt.show()