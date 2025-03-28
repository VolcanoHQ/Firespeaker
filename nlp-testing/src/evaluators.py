# src/evaluators.py
# Import required libraries
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import seaborn as sns
from pathlib import Path
import psutil
import gc  # For garbage collection
from tqdm.notebook import tqdm  # For progress bars

def create_ground_truth(sample_id, text, dialogues=None, characters=None, emotions=None):
    """Create or update ground truth annotations for a text sample"""
    ground_truth = {
        "sample_id": sample_id,
        "text": text,
        "dialogues": dialogues or [],
        "characters": characters or [],
        "emotions": emotions or []
    }
    
    # Save to JSON
    os.makedirs("../data/ground_truth", exist_ok=True)
    with open(f"../data/ground_truth/{sample_id.replace('/', '_')}.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth

def load_ground_truth(sample_id):
    """Load ground truth annotations for a text sample"""
    path = f"../data/ground_truth/{sample_id.replace('/', '_')}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def calculate_metrics(predictions, ground_truth):
    """
    Calculate precision, recall, and F1-score
    
    Parameters:
    - predictions: list of predicted items
    - ground_truth: list of ground truth items
    
    Returns:
    - Dictionary with precision, recall, and F1 metrics
    """
    # Convert to sets for comparison
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    
    # Calculate metrics
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def load_samples(directory="data/corpus"):
    """Load sample texts from corpus directory"""
    samples = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                sample_id = os.path.relpath(file_path, directory)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        samples[sample_id] = f.read()
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return samples

def run_complete_evaluation(benchmark_function, libraries=["spacy", "nltk"], azure_client=None):
    """Run the complete evaluation pipeline on all samples"""
    samples = load_samples()
    results = {}
    
    for sample_id, text in samples.items():
        print(f"Processing sample: {sample_id}")
        
        # Load ground truth if available
        ground_truth_path = f"data/ground_truth/{os.path.basename(sample_id)}.json"
        ground_truth = None
        if os.path.exists(ground_truth_path):
            with open(ground_truth_path, "r") as f:
                ground_truth = json.load(f)
        
        # Run benchmarks
        libs = libraries
        if azure_client is None and "azure" in libs:
            libs.remove("azure")
        
        benchmark_results = benchmark_function(text, task="all", libraries=libs, azure_client=azure_client)
        
        # Calculate metrics against ground truth if available
        evaluation_metrics = {}
        if ground_truth:
            for task in ["dialogue_detection", "ner", "sentiment"]:
                evaluation_metrics[task] = {}
                for lib in benchmark_results[task]:
                    # Extract predictions based on task
                    if task == "dialogue_detection":
                        predictions = benchmark_results[task][lib]["detected_dialogue"]
                        evaluation_metrics[task][lib] = calculate_metrics(predictions, ground_truth["dialogues"])
                    elif task == "ner":
                        predictions = [entity[0] for entity in benchmark_results[task][lib]["entities"]]
                        evaluation_metrics[task][lib] = calculate_metrics(predictions, ground_truth["characters"])
                    # Add sentiment evaluation if needed
        
        results[sample_id] = {
            "benchmark_results": benchmark_results,
            "evaluation_metrics": evaluation_metrics
        }
    
    return results

def generate_summary_report(all_results):
    """Generate a summary report of the evaluation results"""
    summary = {
        "dialogue_detection": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "ner": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "sentiment": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "processing_speed": {lib: {"chars_per_second": []} for lib in ["spacy", "nltk", "azure"]}
    }
    
    # Collect metrics across all samples
    for sample_id, result in all_results.items():
        for task in ["dialogue_detection", "ner", "sentiment"]:
            if task in result["evaluation_metrics"]:
                for lib in result["evaluation_metrics"][task]:
                    metrics = result["evaluation_metrics"][task][lib]
                    for metric in ["precision", "recall", "f1"]:
                        if metric in metrics:
                            summary[task][lib][metric].append(metrics[metric])
        
        # Processing speed
        for lib in result["benchmark_results"]["processing_speed"]:
            summary["processing_speed"][lib]["chars_per_second"].append(
                result["benchmark_results"]["processing_speed"][lib]["chars_per_second"]
            )
    
    # Calculate averages
    for task in summary:
        for lib in summary[task]:
            for metric in summary[task][lib]:
                values = summary[task][lib][metric]
                summary[task][lib][metric] = sum(values) / len(values) if values else 0
    
    return summary