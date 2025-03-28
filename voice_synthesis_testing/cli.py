# voice_synthesis_testing/cli.py
import argparse
import os
import logging
import yaml
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'voice_synthesis_test_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Voice Synthesis Testing Suite')
    
    # Config file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    
    # Command selection
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthesize command
    synth_parser = subparsers.add_parser('synthesize', help='Run text-to-speech synthesis')
    synth_parser.add_argument('--text', type=str, help='Text to synthesize')
    synth_parser.add_argument('--text-file', type=str, help='File containing text to synthesize')
    synth_parser.add_argument('--output-dir', type=str, default='output',
                             help='Directory to save synthesized audio')
    synth_parser.add_argument('--model', type=str, help='Model to use for synthesis')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate synthesized audio')
    eval_parser.add_argument('--reference-dir', type=str, required=True,
                            help='Directory containing reference audio files')
    eval_parser.add_argument('--synthesized-dir', type=str, required=True,
                            help='Directory containing synthesized audio files')
    eval_parser.add_argument('--reference-text', type=str,
                            help='File mapping audio filenames to reference text')
    eval_parser.add_argument('--output-dir', type=str, default='reports',
                            help='Directory to save evaluation reports')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', type=str, nargs='+', required=True,
                               help='List of models to compare')
    compare_parser.add_argument('--text-file', type=str, required=True,
                               help='File containing text for synthesis')
    compare_parser.add_argument('--output-dir', type=str, default='comparison',
                               help='Directory to save comparison results')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--audio-file', type=str, help='Audio file to visualize')
    viz_parser.add_argument('--reference-file', type=str, help='Reference audio file for comparison')
    viz_parser.add_argument('--results-file', type=str, help='Results file to visualize metrics from')
    viz_parser.add_argument('--output-dir', type=str, default='visualizations',
                           help='Directory to save visualizations')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
    batch_parser.add_argument('--input-dir', type=str, required=True,
                             help='Directory containing input files')
    batch_parser.add_argument('--output-dir', type=str, default='batch_output',
                             help='Directory to save batch results')
    batch_parser.add_argument('--model', type=str, help='Model to use for processing')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run automated tests')
    test_parser.add_argument('--test-type', type=str, choices=['unit', 'integration', 'performance'],
                            default='unit', help='Type of tests to run')
    test_parser.add_argument('--test-pattern', type=str, default='test_*.py',
                            help='Pattern to match test files')
    
    # Logging options
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file"""
    try:
        import config as config_module
        if config_path:
            cfg = config_module.load_config(config_path)
        else:
            cfg = config_module.load_config()
            
        # Validate the configuration
        config_module.validate_config(cfg)
        return cfg
    except config_module.ConfigurationError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def run_synthesize(args, config, logger):
    """Run synthesis command"""
    logger.info("Running text-to-speech synthesis")
    
    from voice_synthesis_testing.synthesizer import Synthesizer
    
    synthesizer = Synthesizer(config)
    
    # Get text to synthesize
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        logger.error("No text provided for synthesis. Use --text or --text-file")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Synthesize
    logger.info(f"Synthesizing {len(texts)} text samples using model {args.model or 'default'}")
    output_files = synthesizer.batch_synthesize(
        texts=texts,
        output_dir=args.output_dir,
        model_name=args.model,
        filename_prefix="synth_"
    )
    
    logger.info(f"Synthesis complete. Generated {len(output_files)} audio files in {args.output_dir}")

def run_evaluate(args, config, logger):
    """Run evaluation command"""
    logger.info("Running evaluation")
    
    from voice_synthesis_testing.evaluator import Evaluator
    from voice_synthesis_testing.report_generator import ReportGenerator
    from voice_synthesis_testing.visualizer import Visualizer
    
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    report_gen = ReportGenerator(config)
    
    # Load reference text if provided
    reference_texts = None
    if args.reference_text:
        try:
            with open(args.reference_text, 'r') as f:
                reference_texts = {}
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        filename, text = parts
                        reference_texts[filename] = text
            logger.info(f"Loaded reference text for {len(reference_texts)} files")
        except Exception as e:
            logger.error(f"Error loading reference text: {str(e)}")
    
    # Run evaluation
    results = evaluator.batch_evaluate(
        reference_dir=args.reference_dir,
        synthesized_dir=args.synthesized_dir,
        reference_texts=reference_texts
    )
    
    # Create visualization
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    metrics_plot_path = os.path.join(vis_dir, 'metrics_summary.png')
    visualizer.plot_metrics_summary(results, save_path=metrics_plot_path)
    
    # Generate report
    report_gen.add_test_metadata(
        test_name="Voice Synthesis Evaluation",
        model_name=config.get('model_name', 'Unknown'),
        test_description="Evaluation of synthesized audio against reference"
    )
    
    report_gen.add_metrics(results)
    report_gen.add_plot_path(metrics_plot_path)
    
    # Add sample info
    reference_files = os.listdir(args.reference_dir)
    report_gen.add_sample_info(
        total_samples=len(reference_files),
        sample_rate=config.get('sample_rate', 22050),
        sample_ids=reference_files[:10]  # Include first 10 sample IDs
    )
    
    # Generate report in multiple formats
    report_gen.generate_report(
        output_dir=args.output_dir,
        formats=['json', 'html', 'markdown'],
        base_filename="evaluation_report"
    )
    
    logger.info(f"Evaluation complete. Reports saved to {args.output_dir}")

def run_compare(args, config, logger):
    """Run model comparison command"""
    logger.info(f"Running comparison of models: {', '.join(args.models)}")
    
    from voice_synthesis_testing.synthesizer import Synthesizer
    from voice_synthesis_testing.evaluator import Evaluator
    from voice_synthesis_testing.report_generator import ReportGenerator
    from voice_synthesis_testing.visualizer import Visualizer
    
    synthesizer = Synthesizer(config)
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    report_gen = ReportGenerator(config)
    
    # Load text for synthesis
    with open(args.text_file, 'r') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Loaded {len(texts)} text samples for comparison")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    reference_dir = os.path.join(args.output_dir, 'reference')
    os.makedirs(reference_dir, exist_ok=True)
    
    # Dictionary to store metrics for each model
    model_metrics = {}
    
    # Track performance metrics
    performance_metrics = {}
    
    # Process each model
    for model_name in args.models:
        logger.info(f"Processing model: {model_name}")
        
        # Create model output directory
        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Synthesize with this model
        import time
        import psutil
        
        process = psutil.Process(os.getpid())
        
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        gpu_memory = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Synthesize all texts
        output_files = synthesizer.batch_synthesize(
            texts=texts,
            output_dir=model_dir,
            model_name=model_name,
            filename_prefix=f"{model_name}_"
        )
        
        # Measure performance
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(texts) / duration
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        performance_metrics[model_name] = {
            'duration_seconds': duration,
            'memory_usage_mb': memory_usage,
            'throughput': throughput,
            'gpu_memory_mb': gpu_memory
        }
        
        logger.info(f"Synthesis with {model_name} complete. Generated {len(output_files)} files in {duration:.2f} seconds")
        
        # If this is the first model, use it as reference
        if model_name == args.models[0]:
            # Copy files to reference dir
            import shutil
            for src_file in output_files:
                dest_file = os.path.join(reference_dir, os.path.basename(src_file))
                shutil.copy(src_file, dest_file)
            
            logger.info(f"Using {model_name} as reference model")
            continue
        
        # Evaluate against reference
        logger.info(f"Evaluating {model_name} against reference")
        eval_results = evaluator.batch_evaluate(
            reference_dir=reference_dir,
            synthesized_dir=model_dir
        )
        
        # Store results
        model_metrics[model_name] = eval_results
    
    # Create visualization
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot model comparison
    metrics_to_compare = ['mcd_mean', 'f0_rmse_mean']
    if 'wer_mean' in next(iter(model_metrics.values())):
        metrics_to_compare.append('wer_mean')
    
    comparison_plot_path = os.path.join(vis_dir, 'model_comparison.png')
    visualizer.plot_model_comparison(
        metrics_by_model=model_metrics,
        metric_names=metrics_to_compare,
        title="Model Comparison",
        save_path=comparison_plot_path
    )
    
    # Plot performance comparison
    perf_metrics_to_compare = ['duration_seconds', 'memory_usage_mb', 'throughput']
    if all(m.get('gpu_memory_mb') is not None for m in performance_metrics.values()):
        perf_metrics_to_compare.append('gpu_memory_mb')
    
    perf_comparison_plot_path = os.path.join(vis_dir, 'performance_comparison.png')
    visualizer.plot_model_comparison(
        metrics_by_model=performance_metrics,
        metric_names=perf_metrics_to_compare,
        title="Performance Comparison",
        save_path=perf_comparison_plot_path
    )
    
    # Generate report
    report_gen.add_test_metadata(
        test_name="Voice Synthesis Model Comparison",
        model_name=f"Multiple ({', '.join(args.models)})",
        test_description=f"Comparison of {len(args.models)} voice synthesis models"
    )
    
    report_gen.add_comparison_results(model_metrics)
    report_gen.add_metrics(performance_metrics, category='performance_comparison')
    report_gen.add_plot_path(comparison_plot_path)
    report_gen.add_plot_path(perf_comparison_plot_path)
    
    # Add sample info
    report_gen.add_sample_info(
        total_samples=len(texts),
        sample_rate=config.get('sample_rate', 22050)
    )
    
    # Generate report in multiple formats
    report_gen.generate_report(
        output_dir=args.output_dir,
        formats=['json', 'html', 'markdown'],
        base_filename="model_comparison_report"
    )
    
    logger.info(f"Model comparison complete. Reports saved to {args.output_dir}")

def run_visualize(args, config, logger):
    """Run visualization command"""
    logger.info("Running visualization")
    
    from voice_synthesis_testing.visualizer import Visualizer
    import librosa
    import json
    
    visualizer = Visualizer(config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize audio file if provided
    if args.audio_file:
        audio, sr = librosa.load(args.audio_file, sr=None)
        
        # Generate various visualizations
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        
        # Waveform
        waveform_path = os.path.join(args.output_dir, f"{base_name}_waveform.png")
        visualizer.plot_waveform(
            audio=audio,
            sr=sr,
            title=f"Waveform: {base_name}",
            save_path=waveform_path
        )
        
        # Spectrogram
        spectrogram_path = os.path.join(args.output_dir, f"{base_name}_spectrogram.png")
        visualizer.plot_spectrogram(
            audio=audio,
            sr=sr,
            title=f"Spectrogram: {base_name}",
            save_path=spectrogram_path
        )
        
        # Mel Spectrogram
        mel_path = os.path.join(args.output_dir, f"{base_name}_melspectrogram.png")
        visualizer.plot_melspectrogram(
            audio=audio,
            sr=sr,
            title=f"Mel Spectrogram: {base_name}",
            save_path=mel_path
        )
        
        # F0 Contour
        f0_path = os.path.join(args.output_dir, f"{base_name}_f0.png")
        visualizer.plot_f0_contour(
            audio=audio,
            sr=sr,
            title=f"F0 Contour: {base_name}",
            save_path=f0_path
        )
        
        logger.info(f"Generated visualizations for {args.audio_file}")
    
    # Compare with reference if provided
    if args.audio_file and args.reference_file:
        audio, sr = librosa.load(args.audio_file, sr=None)
        ref_audio, ref_sr = librosa.load(args.reference_file, sr=None)
        
        # Resample if needed
        if sr != ref_sr:
            logger.info(f"Resampling from {sr} to {ref_sr}")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ref_sr)
            sr = ref_sr
        
        # Generate comparison
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        ref_name = os.path.splitext(os.path.basename(args.reference_file))[0]
        
        comparison_path = os.path.join(args.output_dir, f"{base_name}_vs_{ref_name}.png")
        
        # Calculate metrics if evaluator is available
        metrics = None
        try:
            from voice_synthesis_testing.evaluator import Evaluator
            evaluator = Evaluator(config)
            
            metrics = {
                'mcd': evaluator.calculate_mcd(ref_audio, audio, sr),
                'f0_rmse': evaluator.calculate_f0_rmse(ref_audio, audio, sr)
            }
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {str(e)}")
        
        visualizer.plot_comparison(
            ref_audio=ref_audio,
            synth_audio=audio,
            sr=sr,
            metrics=metrics,
            title=f"Comparison: {ref_name} vs {base_name}",
            save_path=comparison_path
        )
        
        logger.info(f"Generated comparison visualization")
    
    # Visualize results file if provided
    if args.results_file:
        try:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            # Determine type of results file
            if 'model_comparison' in results:
                # Model comparison results
                model_metrics = results['model_comparison']
                
                metrics_to_compare = ['mcd_mean', 'f0_rmse_mean']
                if 'wer_mean' in next(iter(model_metrics.values())):
                    metrics_to_compare.append('wer_mean')
                
                comparison_path = os.path.join(args.output_dir, "model_comparison.png")
                visualizer.plot_model_comparison(
                    metrics_by_model=model_metrics,
                    metric_names=metrics_to_compare,
                    title="Model Comparison",
                    save_path=comparison_path
                )
                
                logger.info(f"Generated model comparison visualization")
                
            elif 'quality_metrics' in results:
                # Single evaluation results
                metrics = results['quality_metrics']
                
                # Extract list metrics
                list_metrics = {k: v for k, v in metrics.items() 
                               if isinstance(v, list) and not any(suffix in k for suffix in 
                                                                ['_mean', '_std', '_min', '_max'])}
                
                if list_metrics:
                    metrics_path = os.path.join(args.output_dir, "metrics_summary.png")
                    visualizer.plot_metrics_summary(
                        metrics=list_metrics,
                        title="Evaluation Metrics Summary",
                        save_path=metrics_path
                    )
                    
                    logger.info(f"Generated metrics summary visualization")
                else:
                    logger.warning("No list metrics found in results file")
            else:
                logger.warning("Unrecognized results format")
                
        except Exception as e:
            logger.error(f"Error visualizing results file: {str(e)}")
    
    logger.info(f"Visualization complete. Results saved to {args.output_dir}")

def run_batch(args, config, logger):
    """Run batch processing command"""
    logger.info("Running batch processing")
    
    # Determine input files type based on directory content
    input_files = sorted(os.listdir(args.input_dir))
    
    if not input_files:
        logger.error(f"No files found in input directory: {args.input_dir}")
        return
    
    # Check file extensions to determine processing type
    first_file = input_files[0]
    file_ext = os.path.splitext(first_file)[1].lower()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if file_ext in ['.txt', '.csv']:
        # Text files - treat as synthesis batch
        logger.info(f"Detected text files for batch synthesis")
        
        from voice_synthesis_testing.synthesizer import Synthesizer
        synthesizer = Synthesizer(config)
        
        # Collect all texts
        all_texts = []
        for file_name in input_files:
            if not file_name.endswith(file_ext):
                continue
                
            file_path = os.path.join(args.input_dir, file_name)
            with open(file_path, 'r') as f:
                if file_ext == '.txt':
                    # Each line is a text to synthesize
                    texts = [line.strip() for line in f.readlines() if line.strip()]
                else:  # .csv
                    # Assume CSV has text in the first column
                    import csv
                    texts = []
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0].strip():
                            texts.append(row[0].strip())
                
                all_texts.extend(texts)
        
        # Synthesize
        logger.info(f"Synthesizing {len(all_texts)} text samples using model {args.model or 'default'}")
        output_files = synthesizer.batch_synthesize(
            texts=all_texts,
            output_dir=args.output_dir,
            model_name=args.model,
            filename_prefix="batch_"
        )
        
        logger.info(f"Batch synthesis complete. Generated {len(output_files)} audio files in {args.output_dir}")
        
    elif file_ext in ['.wav', '.mp3', '.flac', '.ogg']:
        # Audio files - treat as feature extraction batch
        logger.info(f"Detected audio files for batch feature extraction")
        
        from voice_synthesis_testing.feature_processor import FeatureProcessor
        feature_processor = FeatureProcessor(config)
        
        # Process files
        file_paths = [os.path.join(args.input_dir, f) for f in input_files 
                     if f.endswith(file_ext)]
        
        # Extract features
        logger.info(f"Extracting features from {len(file_paths)} audio files")
        features = feature_processor.batch_process_files(file_paths)
        
        # Save features
        import numpy as np
        for i, file_path in enumerate(file_paths):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            feature_dir = os.path.join(args.output_dir, base_name)
            os.makedirs(feature_dir, exist_ok=True)
            
            for feat_name, feat_list in features.items():
                if i < len(feat_list):
                    feat_data = feat_list[i]
                    feat_path = os.path.join(feature_dir, f"{feat_name}.npy")
                    np.save(feat_path, feat_data)
        
        logger.info(f"Batch feature extraction complete. Results saved to {args.output_dir}")
        
    else:
        logger.error(f"Unsupported file type for batch processing: {file_ext}")
        
def run_tests(args, config, logger):
    """Run automated tests"""
    logger.info(f"Running {args.test_type} tests")
    
    import unittest
    import sys
    
    # Determine test directory
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', args.test_type)
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=args.test_pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    logger.info(f"Tests complete: {result.testsRun} run, {len(result.errors)} errors, {len(result.failures)} failures")
    
    # Exit with appropriate code
    if result.wasSuccessful():
        return 0
    else:
        return 1

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(args.log_dir, log_level)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # For test commands, create a test-specific configuration
    if args.command in ['synthesize', 'evaluate', 'compare']:
        import config as config_module
        test_name = f"{args.command}_test"
        config_dict = config_module.get_test_config(config_dict, test_name)
        logger.info(f"Created test-specific configuration: {test_name}")
    
    # Create Namespace object from config dictionary for easier access
    class ConfigNamespace:
        def __init__(self, config_dict):
            self.__dict__.update(config_dict)
    
    config_obj = ConfigNamespace(config_dict)
    
    logger.info(f"Starting Voice Synthesis Testing Suite")
    logger.info(f"Command: {args.command}")
    
    # Execute selected command
    try:
        if args.command == 'synthesize':
            run_synthesize(args, config_obj, logger)
        elif args.command == 'evaluate':
            run_evaluate(args, config_obj, logger)
        elif args.command == 'compare':
            run_compare(args, config_obj, logger)
        elif args.command == 'visualize':
            run_visualize(args, config_obj, logger)
        elif args.command == 'batch':
            run_batch(args, config_obj, logger)
        elif args.command == 'test':
            return run_tests(args, config_obj, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
        return 0
    except Exception as e:
        logger.error(f"Error executing command {args.command}: {str(e)}", exc_info=True)
        return 1
    
if __name__ == "__main__":
    sys.exit(main())