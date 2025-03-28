# voice_synthesis_testing/report_generator.py
import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    """
    Generates structured test reports in various formats.
    """
    
    def __init__(self, config):
        """
        Initialize the report generator with configuration
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.report_data = {}
    
    def add_test_metadata(self, test_name: str, model_name: str, 
                         test_date: Optional[datetime] = None,
                         test_description: Optional[str] = None,
                         hardware_info: Optional[Dict[str, str]] = None):
        """
        Add metadata about the test
        
        Args:
            test_name: Name of the test
            model_name: Name of the model being tested
            test_date: Date of the test (defaults to now)
            test_description: Description of the test
            hardware_info: Information about the hardware used for testing
        """
        if test_date is None:
            test_date = datetime.now()
            
        self.report_data['metadata'] = {
            'test_name': test_name,
            'model_name': model_name,
            'test_date': test_date.strftime('%Y-%m-%d %H:%M:%S'),
            'test_description': test_description or ''
        }
        
        # Add hardware info if provided
        if hardware_info:
            self.report_data['metadata']['hardware_info'] = hardware_info
    
    def add_metrics(self, metrics: Dict[str, Any], category: str = 'quality_metrics'):
        """
        Add evaluation metrics to the report
        
        Args:
            metrics: Dictionary of metrics
            category: Category to organize metrics under
        """
        if category not in self.report_data:
            self.report_data[category] = {}
            
        self.report_data[category].update(metrics)
    
    def add_performance_metrics(self, duration_seconds: float, 
                               memory_usage_mb: float,
                               throughput: Optional[float] = None,
                               gpu_memory_mb: Optional[float] = None):
        """
        Add performance metrics to the report
        
        Args:
            duration_seconds: Total processing time in seconds
            memory_usage_mb: Memory usage in MB
            throughput: Speed metric (e.g., samples per second)
            gpu_memory_mb: GPU memory usage if applicable
        """
        perf_metrics = {
            'duration_seconds': duration_seconds,
            'memory_usage_mb': memory_usage_mb
        }
        
        if throughput is not None:
            perf_metrics['throughput'] = throughput
            
        if gpu_memory_mb is not None:
            perf_metrics['gpu_memory_mb'] = gpu_memory_mb
            
        self.add_metrics(perf_metrics, category='performance_metrics')
    
    def add_sample_info(self, total_samples: int, 
                       sample_rate: int,
                       sample_ids: Optional[List[str]] = None):
        """
        Add information about the test samples
        
        Args:
            total_samples: Total number of samples tested
            sample_rate: Sample rate of audio
            sample_ids: List of sample IDs (optional)
        """
        sample_info = {
            'total_samples': total_samples,
            'sample_rate': sample_rate
        }
        
        if sample_ids:
            sample_info['sample_ids'] = sample_ids
            
        self.add_metrics(sample_info, category='sample_info')
    
    def add_config_info(self, model_config: Dict[str, Any]):
        """
        Add information about the model configuration
        
        Args:
            model_config: Dictionary of model configuration parameters
        """
        self.add_metrics(model_config, category='model_config')
    
    def add_comparison_results(self, comparison_data: Dict[str, Dict[str, Any]]):
        """
        Add results from model comparison
        
        Args:
            comparison_data: Dictionary mapping model names to their metrics
        """
        self.add_metrics(comparison_data, category='model_comparison')
    
    def save_json(self, output_path: str):
        """
        Save report as JSON
        
        Args:
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle datetime objects for JSON serialization
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=json_serial)
            
        self.logger.info(f"Saved JSON report to {output_path}")
    
    def save_yaml(self, output_path: str):
        """
        Save report as YAML
        
        Args:
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert NumPy types to Python native types for YAML
        def convert_numpy(item):
            if isinstance(item, dict):
                return {k: convert_numpy(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_numpy(i) for i in item]
            elif isinstance(item, np.integer):
                return int(item)
            elif isinstance(item, np.floating):
                return float(item)
            elif isinstance(item, np.ndarray):
                return convert_numpy(item.tolist())
            else:
                return item
        
        with open(output_path, 'w') as f:
            yaml.dump(convert_numpy(self.report_data), f, default_flow_style=False)
            
        self.logger.info(f"Saved YAML report to {output_path}")
    
    def save_csv(self, output_path: str):
        """
        Save metrics as CSV (flattens the structure)
        
        Args:
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Flatten the nested dictionary structure
        flat_data = {}
        
        for category, metrics in self.report_data.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    # Skip lists and dicts, only include scalar values
                    if not isinstance(value, (list, dict)):
                        flat_data[f"{category}_{metric_name}"] = value
        
        # Convert to DataFrame for easy CSV export
        df = pd.DataFrame([flat_data])
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Saved CSV report to {output_path}")
    
    def save_markdown(self, output_path: str):
        """
        Save report as Markdown
        
        Args:
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Title
            if 'metadata' in self.report_data and 'test_name' in self.report_data['metadata']:
                f.write(f"# {self.report_data['metadata']['test_name']}\n\n")
            else:
                f.write("# Voice Synthesis Test Report\n\n")
                
            # Metadata
            if 'metadata' in self.report_data:
                f.write("## Metadata\n\n")
                for key, value in self.report_data['metadata'].items():
                    if isinstance(value, dict):
                        f.write(f"### {key.replace('_', ' ').title()}\n\n")
                        for subkey, subvalue in value.items():
                            f.write(f"- **{subkey.replace('_', ' ').title()}**: {subvalue}\n")
                        f.write("\n")
                    else:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")
            
            # Other sections
            for category, metrics in self.report_data.items():
                if category != 'metadata' and isinstance(metrics, dict):
                    f.write(f"## {category.replace('_', ' ').title()}\n\n")
                    
                    # Handle different types of metrics
                    scalar_metrics = {}
                    list_metrics = {}
                    dict_metrics = {}
                    
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            scalar_metrics[metric_name] = value
                        elif isinstance(value, list):
                            list_metrics[metric_name] = value
                        elif isinstance(value, dict):
                            dict_metrics[metric_name] = value
                    
                    # Write scalar metrics as a table
                    if scalar_metrics:
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for metric_name, value in scalar_metrics.items():
                            f.write(f"| {metric_name.replace('_', ' ').title()} | {value} |\n")
                        f.write("\n")
                    
                    # Handle list metrics
                    for metric_name, values in list_metrics.items():
                        f.write(f"### {metric_name.replace('_', ' ').title()}\n\n")
                        if len(values) <= 10:  # For short lists
                            for i, value in enumerate(values):
                                f.write(f"- Sample {i}: {value}\n")
                        else:  # For longer lists, show summary stats
                            f.write(f"- Count: {len(values)}\n")
                            f.write(f"- Mean: {np.mean(values)}\n")
                            f.write(f"- Min: {np.min(values)}\n")
                            f.write(f"- Max: {np.max(values)}\n")
                            f.write(f"- Std: {np.std(values)}\n")
                        f.write("\n")
                    
                    # Handle nested dictionaries
                    for metric_name, sub_metrics in dict_metrics.items():
                        f.write(f"### {metric_name.replace('_', ' ').title()}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for sub_name, sub_value in sub_metrics.items():
                            if not isinstance(sub_value, (list, dict)):
                                f.write(f"| {sub_name.replace('_', ' ').title()} | {sub_value} |\n")
                        f.write("\n")
            
            # Footer
            f.write("---\n")
            f.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        self.logger.info(f"Saved Markdown report to {output_path}")
    
    def save_html(self, output_path: str, include_plots: bool = True):
        """
        Save report as HTML
        
        Args:
            output_path: Path to save the report
            include_plots: Whether to include plots
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            from jinja2 import Template
            
            # Basic HTML template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    h1, h2, h3 { color: #333; }
                    .plot-container { margin: 20px 0; text-align: center; }
                    img { max-width: 100%; }
                </style>
            </head>
            <body>
                <h1>{{ title }}</h1>
                
                {% if metadata %}
                <h2>Metadata</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    {% for key, value in metadata.items() %}
                    {% if not isinstance(value, dict) %}
                    <tr><td>{{ key | replace('_', ' ') | title }}</td><td>{{ value }}</td></tr>
                    {% endif %}
                    {% endfor %}
                </table>
                
                {% for key, value in metadata.items() %}
                {% if isinstance(value, dict) %}
                <h3>{{ key | replace('_', ' ') | title }}</h3>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    {% for subkey, subvalue in value.items() %}
                    <tr><td>{{ subkey | replace('_', ' ') | title }}</td><td>{{ subvalue }}</td></tr>
                    {% endfor %}
                </table>
                {% endif %}
                {% endfor %}
                {% endif %}
                
                {% for category, metrics in data.items() %}
                {% if category != 'metadata' and isinstance(metrics, dict) %}
                <h2>{{ category | replace('_', ' ') | title }}</h2>
                
                {% set scalar_metrics = {} %}
                {% set list_metrics = {} %}
                {% set dict_metrics = {} %}
                
                {% for metric_name, value in metrics.items() %}
                {% if isinstance(value, (int, float, str, bool)) %}
                {% set _ = scalar_metrics.update({metric_name: value}) %}
                {% elif isinstance(value, list) %}
                {% set _ = list_metrics.update({metric_name: value}) %}
                {% elif isinstance(value, dict) %}
                {% set _ = dict_metrics.update({metric_name: value}) %}
                {% endif %}
                {% endfor %}
                
                {% if scalar_metrics %}
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {% for metric_name, value in scalar_metrics.items() %}
                    <tr><td>{{ metric_name | replace('_', ' ') | title }}</td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
                {% endif %}
                
                {% for metric_name, values in list_metrics.items() %}
                <h3>{{ metric_name | replace('_', ' ') | title }}</h3>
                {% if values | length <= 10 %}
                <ul>
                    {% for i, value in enumerate(values) %}
                    <li>Sample {{ i }}: {{ value }}</li>
                    {% endfor %}
                </ul>
                {% else %}
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Count</td><td>{{ values | length }}</td></tr>
                    <tr><td>Mean</td><td>{{ values | mean }}</td></tr>
                    <tr><td>Min</td><td>{{ values | min }}</td></tr>
                    <tr><td>Max</td><td>{{ values | max }}</td></tr>
                    <tr><td>Std</td><td>{{ values | std }}</td></tr>
                </table>
                {% endif %}
                {% endfor %}
                
                {% for metric_name, sub_metrics in dict_metrics.items() %}
                <h3>{{ metric_name | replace('_', ' ') | title }}</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {% for sub_name, sub_value in sub_metrics.items() %}
                    {% if not isinstance(sub_value, (list, dict)) %}
                    <tr><td>{{ sub_name | replace('_', ' ') | title }}</td><td>{{ sub_value }}</td></tr>
                    {% endif %}
                    {% endfor %}
                </table>
                {% endfor %}
                
                {% endif %}
                {% endfor %}
                
                {% if plots %}
                <h2>Visualizations</h2>
                {% for plot in plots %}
                <div class="plot-container">
                    <h3>{{ plot.title }}</h3>
                    <img src="{{ plot.path }}" alt="{{ plot.title }}">
                </div>
                {% endfor %}
                {% endif %}
                
                <hr>
                <p>Report generated on {{ timestamp }}</p>
            </body>
            </html>
            """
            
            # Custom filters for template
            def is_instance(value, cls):
                return isinstance(value, cls)
                
            def mean(values):
                return np.mean(values) if values else 0
                
            def min_val(values):
                return np.min(values) if values else 0
                
            def max_val(values):
                return np.max(values) if values else 0
                
            def std(values):
                return np.std(values) if values else 0
                
            template = Template(template_str)
            template.globals['isinstance'] = isinstance
            template.globals['mean'] = mean
            template.globals['min'] = min_val
            template.globals['max'] = max_val
            template.globals['std'] = std
            
            # Prepare data for template
            title = self.report_data.get('metadata', {}).get('test_name', 'Voice Synthesis Test Report')
            
            # Get plots if included
            plots = []
            if include_plots and 'plot_paths' in self.report_data:
                for plot_path in self.report_data['plot_paths']:
                    # Extract file name as title
                    plot_title = os.path.splitext(os.path.basename(plot_path))[0].replace('_', ' ').title()
                    plots.append({
                        'title': plot_title,
                        'path': plot_path
                    })
            
            # Render HTML
            html = template.render(
                title=title,
                metadata=self.report_data.get('metadata', {}),
                data=self.report_data,
                plots=plots,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html)
                
            self.logger.info(f"Saved HTML report to {output_path}")
                
        except ImportError:
            self.logger.warning("Jinja2 not installed. Falling back to Markdown report.")
            md_path = output_path.replace('.html', '.md')
            self.save_markdown(md_path)
    
    def add_plot_path(self, plot_path: str):
        """
        Add a path to a generated plot
        
        Args:
            plot_path: Path to the plot file
        """
        if 'plot_paths' not in self.report_data:
            self.report_data['plot_paths'] = []
            
        self.report_data['plot_paths'].append(plot_path)
    
    def generate_report(self, output_dir: str, 
                       formats: List[str] = ['json', 'html'],
                       base_filename: Optional[str] = None):
        """
        Generate reports in multiple formats
        
        Args:
            output_dir: Directory to save reports
            formats: List of formats to generate (json, yaml, csv, markdown, html)
            base_filename: Base filename (defaults to test_name or 'voice_synthesis_report')
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine base filename
        if base_filename is None:
            if 'metadata' in self.report_data and 'test_name' in self.report_data['metadata']:
                base_filename = self.report_data['metadata']['test_name'].lower().replace(' ', '_')
            else:
                base_filename = 'voice_synthesis_report'
                
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_filename}_{timestamp}"
        
        # Generate each requested format
        for format_type in formats:
            if format_type.lower() == 'json':
                self.save_json(os.path.join(output_dir, f"{filename}.json"))
            elif format_type.lower() == 'yaml':
                self.save_yaml(os.path.join(output_dir, f"{filename}.yaml"))
            elif format_type.lower() == 'csv':
                self.save_csv(os.path.join(output_dir, f"{filename}.csv"))
            elif format_type.lower() == 'markdown':
                self.save_markdown(os.path.join(output_dir, f"{filename}.md"))
            elif format_type.lower() == 'html':
                self.save_html(os.path.join(output_dir, f"{filename}.html"))
            else:
                self.logger.warning(f"Unsupported format: {format_type}")
                
        self.logger.info(f"Generated reports in formats: {', '.join(formats)}")