"""
This script benchmarks the performance of NLTK and spaCy libraries on tasks relevant to audiobook creation. 
It includes functionalities for Named Entity Recognition (NER), Sentiment Analysis, Dialogue Detection, 
and Processing Speed evaluation. The script supports parallel processing, chunking for large files, 
result aggregation, visualization, and basic ground truth validation.
Features:
- Named Entity Recognition (Characters): Identifies character names in the text.
- Sentiment Analysis: Analyzes the sentiment polarity and subjectivity of sentences.
- Dialogue Detection: Detects dialogue sentences enclosed in quotes.
- Processing Speed: Measures the speed of text processing for each library.
Key Components:
1. Configuration Constants:
    - Paths for data, results, and ground truth directories.
    - Processing parameters like chunk size, overlap, and maximum workers.
    - Task-specific thresholds for sentiment analysis.
2. Directory and Logging Setup:
    - Ensures necessary directories exist.
    - Configures logging for both file and console outputs.
3. NLTK and spaCy Initialization:
    - Downloads required NLTK resources.
    - Initializes spaCy models with GPU preference and additional pipelines like `spacytextblob`.
4. Task Functions:
    - Implements NER, sentiment analysis, dialogue detection, and speed evaluation for both NLTK and spaCy.
    - Uses decorators for consistent error handling and timing.
5. Benchmarking and Processing:
    - Processes text samples in chunks for large files.
    - Merges chunk results and validates against ground truth data.
6. Analysis and Visualization:
    - Analyzes benchmarking results and generates performance summaries.
    - Creates visualizations for processing time, speed, and task performance.
7. Ground Truth Management:
    - Creates and loads ground truth data for validation.
8. Cleanup and Resource Management:
    - Removes temporary files, closes database connections, and releases resources.
Usage:
- Run the script directly to benchmark NLP tasks on text samples.
- Ensure the required NLTK and spaCy models are installed.
- Customize the configuration constants as needed for specific use cases.
Dependencies:
- Python standard libraries: datetime, os, time, json, logging, gc, traceback, shutil, pathlib, collections, multiprocessing.
- Third-party libraries: nltk, spacy, spacytextblob, numpy, pandas, seaborn, matplotlib, tqdm.
- The script includes robust error handling and logging for debugging and monitoring.
- Ensure sufficient system resources for parallel processing and large text files.
NLP Benchmark Script for Audiobook Feature Extraction
Author: Timmothy Escolopio
"""
import datetime
import sys
import os
import time
import json
import logging
import gc
import re
import traceback # For detailed error logging
from typing import Any, Dict, List, Optional, Tuple, Union # Improved type hinting
import pandas as pd  # Ensure pandas is imported
import matplotlib.pyplot as plt  # Ensure matplotlib is imported
import seaborn as sns  # Ensure seaborn is imported

# Third-party Libraries
# (Add try-except blocks for imports as in the previous version for robustness)
import nltk # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from typing import Dict, Any 
from spacy.matcher import Matcher
from spacytextblob.spacytextblob import SpacyTextBlob # type: ignore
import numpy as np

# Standard Library
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import warnings # To potentially filter warnings if needed
import multiprocessing as mp # Import multiprocessing

# --- Configuration Constants ---
BASE_DIR = Path(__file__).resolve().parent # Assumes script is run directly
DATA_DIR = BASE_DIR / "../data"
CORPUS_DIR = DATA_DIR / "corpus"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = BASE_DIR / "../results"
LOG_FILE = RESULTS_DIR / "benchmark.log" # Place log in results

# Processing Parameters
CHUNK_SIZE = 10000  # Characters per chunk
OVERLAP = 1000      # Character overlap between chunks
MAX_WORKERS = 8 # Default to CPU count or 4
SPACY_MODEL = "en_core_web_trf" # Transformer model for better accuracy
# SPACY_MODEL = "en_core_web_lg" # Large model (faster, less accurate than trf)
# SPACY_MODEL = "en_core_web_sm" # Small model (fastest, least accurate)

# Task-specific Parameters
SENTIMENT_POLARITY_THRESHOLD = 0.1
SENTIMENT_SUBJECTIVITY_THRESHOLD = 0.3
TOP_N_EXAMPLES = 5 # How many example results to store (dialogue, sentiment, NER)

# --- Directory Setup ---
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True) # Ensure GT dir exists

# Define dialogue verbs (lemmas)
DIALOGUE_VERBS = ['say', 'ask', 'reply', 'shout', 'whisper', 'exclaim', 'tell', 'answer', 'add', 'continue', 'begin', 'mutter', 'murmur']

DIALOGUE_VERBS_NLTK = [
    'say', 'said', 'ask', 'asked', 'reply', 'replied', 'shout', 'shouted',
    'whisper', 'whispered', 'exclaim', 'exclaimed', 'tell', 'told',
    'answer', 'answered', 'add', 'added', 'continue', 'continued', 'begin', 'began',
    'mutter', 'muttered', 'murmur', 'murmured'
]
# --- Global Variables ---
# Global spaCy model instance *per worker process* - Initialized in init_worker
nlp_lg: Optional[spacy.language.Language] = None
# Global NLTK sentiment analyzer *per worker process* - Initialized in init_worker
sentiment_analyzer: Optional[SentimentIntensityAnalyzer] = None

# --- Logging Setup ---
logger = logging.getLogger("nlp_benchmark")
logger.setLevel(logging.INFO)
# Define logger instance (ensure it's configured elsewhere)
ana_logger = logging.getLogger("nlp_benchmark")


# Prevent duplicate handlers if script is run multiple times in same session
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(funcName)s - %(message)s" # Added funcName
    )
    # File handler
    fh = logging.FileHandler(LOG_FILE, mode='w') # Overwrite log each run
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- NLTK Downloads ---
# Moved definition here, call inside main guard
def download_nltk_resources():
    """Downloads necessary NLTK data packages if not already present."""
    required_nltk_packages = {
        'tokenizers/punkt': 'punkt',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
        'corpora/vader_lexicon': 'vader_lexicon',
        'taggers/maxent_ne_chunker': 'maxent_ne_chunker',
        'corpora/words': 'words',
    }
    logger.info("Checking NLTK resources...")
    for resource_path, package_id in required_nltk_packages.items():
        try:
            nltk.data.find(resource_path)
            logger.debug(f"NLTK resource '{package_id}' found.")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {package_id}")
            try:
                # Download non-quietly to see progress/errors
                nltk.download(package_id, quiet=False)
                logger.info(f"Successfully downloaded {package_id}")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {package_id}: {e}", exc_info=True)
                # Consider if this is fatal
                # raise RuntimeError(f"Failed to download essential NLTK resource: {package_id}") from e

# --- Global Variables ---
# Global spaCy model instance *per worker process* - Initialized in init_worker
nlp_lg: Optional[spacy.language.Language] = None
# Global NLTK sentiment analyzer *per worker process* - Initialized in init_worker
sentiment_analyzer: Optional[SentimentIntensityAnalyzer] = None

# --- Helper Functions ---
def clear_memory():
    """Explicitly suggest garbage collection."""
    logger.debug("Suggesting garbage collection.")
    gc.collect()

def format_error(e: Exception) -> str:
    """Formats an exception for logging/storage."""
    return f"{type(e).__name__}: {str(e)}"


# --- Worker Initialization (with Enhanced Logging/Checks) ---
def init_worker():
    """
    Initialize the worker process environment by setting up necessary NLP models and tools.
    This function is responsible for:
    - Setting up a logger for the worker process.
    - Initializing the NLTK SentimentIntensityAnalyzer for sentiment analysis.
    - Configuring and loading the spaCy NLP model, with GPU preference if available.
    - Adding the `spacytextblob` pipeline component to the spaCy model for text blob sentiment analysis.
    Key Features:
    - Handles GPU preference and availability checks for spaCy.
    - Logs detailed information about the initialization process, including success and failure cases.
    - Ensures the worker process can continue even if certain components fail to initialize (e.g., NLTK sentiment analyzer).
    Raises:
        RuntimeError: If the spaCy model fails to load due to an OS error (e.g., model not downloaded).
        Exception: Re-raises any general exceptions encountered during spaCy initialization.
    Notes:
    - The function uses global variables `nlp_lg` and `sentiment_analyzer` to store the initialized spaCy model
    and NLTK SentimentIntensityAnalyzer, respectively.
    - Ensure that the required spaCy model (`SPACY_MODEL`) and `spacytextblob` package are installed prior to execution.
    """
    # Get logger instance for this worker
    worker_logger = logging.getLogger("nlp_benchmark")
    # Note: Log level is inherited from the main process config unless reset here
    worker_logger.info("Worker process started. Initializing...") # First log message

    global nlp_lg, sentiment_analyzer
    worker_pid = os.getpid()  # Ensure os is imported and used correctly (no changes needed here)

    # Initialize NLTK Sentiment Analyzer First (less likely to fail)
    try:
        worker_logger.info("Initializing NLTK SentimentIntensityAnalyzer...")
        sentiment_analyzer = SentimentIntensityAnalyzer()
        worker_logger.info("NLTK SentimentIntensityAnalyzer initialized successfully.")
    except Exception as e:
        worker_logger.error("Error initializing NLTK Sentiment Analyzer", exc_info=True)
        sentiment_analyzer = None # Allow processing to continue without NLTK sentiment

    # Initialize spaCy model (More complex part)
    try:
        worker_logger.info("Attempting spaCy GPU preference...")
        try:
            # Set CUDA device visibility if needed (advanced)
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id % num_gpus)
            spacy.prefer_gpu()
            worker_logger.info("spacy.prefer_gpu() called successfully.")
            # Explicitly check if GPU is usable by spaCy *now*
            if spacy.require_gpu():
                worker_logger.info("spacy.require_gpu() confirmed GPU availability.")
            else:
                # This case should ideally not happen if prefer_gpu succeeded without error
                # but reflects that require_gpu checks more deeply.
                worker_logger.warning("spacy.require_gpu() returned False despite prefer_gpu call.")
        except Exception as gpu_err:
            worker_logger.warning(f"GPU preference/requirement check failed: {gpu_err}. Will proceed (likely on CPU).")

        worker_logger.info(f"Loading spaCy model: {SPACY_MODEL}...")
        # Filter the PyTorch warning if desired
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
            # Load the model
            nlp_lg = spacy.load(SPACY_MODEL) # Load with default components for the model
        worker_logger.info(f"spaCy model '{SPACY_MODEL}' LOADED successfully.") # **** CRITICAL LOG ****

        # Check device usage *after* loading
        if hasattr(spacy.util, 'is_gpu_available') and spacy.util.is_gpu_available():
            # Use internal spaCy check first
            worker_logger.info("spaCy reports GPU is available via spacy.util.is_gpu_available().")
            # Check model's pipe devices if possible (more granular)
            if nlp_lg and nlp_lg.pipe_names:
                pipe_devices = {pipe_name: nlp_lg.get_pipe_meta(pipe_name).device for pipe_name in nlp_lg.pipe_names if hasattr(nlp_lg.get_pipe_meta(pipe_name), 'device')}
                worker_logger.info(f"Model pipe devices: {pipe_devices}")
                # Generally, if transformer is on GPU, it's using GPU. -1 means CPU.
                if 'transformer' in pipe_devices and pipe_devices['transformer'] != -1:
                    worker_logger.info(f"Transformer component appears to be on GPU device {pipe_devices['transformer']}.")
                else:
                    worker_logger.warning("Transformer component appears to be on CPU (device -1) or device info unavailable.")
            else:
                worker_logger.info("Could not determine specific pipe devices.")
        else:
            worker_logger.info("spaCy reports GPU is NOT available via spacy.util.is_gpu_available() (using CPU).")


        worker_logger.info("Adding spacytextblob pipe...")
        if "spacytextblob" not in nlp_lg.pipe_names:
            try:
                from spacytextblob.spacytextblob import SpacyTextBlob  # Ensure spacytextblob is imported
                nlp_lg.add_pipe("spacytextblob")
                worker_logger.info("Added spacytextblob to spaCy pipeline.")
            except ImportError:
                worker_logger.error("spacytextblob is not installed. Please install it using 'pip install spacytextblob'.")
        else:
            worker_logger.info("spacytextblob pipe already exists.")
    except OSError as e:
        # Specific error for model not found
        worker_logger.error(f"Fatal OS Error loading spaCy model '{SPACY_MODEL}'. Is it downloaded? (python -m spacy download {SPACY_MODEL})", exc_info=True)
        raise RuntimeError(f"Worker {worker_pid} failed to load spaCy model") from e
    except Exception as e:
        # Catch any other loading error
        worker_logger.error(f"General Error during spaCy initialization", exc_info=True)
        raise # Re-raise the exception to make the worker fail

    worker_logger.info("Worker initialization function finished.")

# --- Task Functions ---
# ner_spacy, ner_nltk, sentiment_spacy, sentiment_nltk,
# speed_spacy, speed_nltk, dialogue_detection_spacy, dialogue_detection_nltk
# benchmark_performance
# _process_chunk_or_sample
# merge_chunk_results
# process_sample
# load_samples
# process_multiple_samples (Ensure it uses the updated init_worker)
# create_ground_truth, load_ground_truth
# analyze_and_save_results, create_visualizations, validate_against_ground_truth
# remove_temp_files, close_database_connections, release_resources
# (No changes needed to the core logic of these functions based on the current error)
# --- Keep the refined versions from the previous response for all these ---
# --- [Paste the previously refined functions here] ---
# Decorator for consistent error handling and timing
def benchmark_task(func):
    """
    A decorator to benchmark a function's execution time and handle errors gracefully.

    This decorator logs the execution time of the wrapped function and captures any errors 
    that occur during its execution. It returns a dictionary containing the execution time, 
    result count, and any error information.

    Args:
        func (Callable): The function to be wrapped and benchmarked.

    Returns:
        Callable: A wrapped function that benchmarks execution time and handles errors.

    The returned dictionary contains the following keys:
        - "time_taken" (float): The time taken to execute the function in seconds.
        - "count" (int): A placeholder for the result count (default is 0).
        - "error" (str or None): A string describing the error if one occurred, otherwise None.

    Raises:
        TypeError: If the first argument to the wrapped function is not a string.
        ValueError: If a value-related error occurs in the wrapped function.
        LookupError: If an NLTK resource is missing.
        RuntimeError: If a runtime error occurs.
        Exception: For any other unexpected errors.

    Example:
        @benchmark_task
        def process_text(text: str) -> Dict[str, Any]:
            # Function implementation here
            pass
    """
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        task_logger = logging.getLogger("nlp_benchmark") # Get logger within task scope
        start_time = time.monotonic()
        result_base = {"time_taken": 0.0, "count": 0, "error": None}
        try:
            if args and not isinstance(args[0], str):
                raise TypeError(f"Input 'text' must be a string, got {type(args[0])}")
            result = func(*args, **kwargs)
            result_base.update(result)
        except (TypeError, ValueError) as e:
            result_base["error"] = format_error(e)
            task_logger.error(f"Input error in {func.__name__}: {result_base['error']}")
        except LookupError as e:
            result_base["error"] = format_error(e)
            task_logger.error(f"NLTK resource missing in {func.__name__}: {result_base['error']}")
        except RuntimeError as e:
            result_base["error"] = format_error(e)
            task_logger.error(f"Runtime error in {func.__name__}: {result_base['error']}")
        except Exception as e:
            result_base["error"] = format_error(e)
            task_logger.error(f"Unexpected error in {func.__name__}: {result_base['error']}", exc_info=True)
        finally:
            result_base["time_taken"] = time.monotonic() - start_time
            if result_base.get("error") is None:
                result_base["error"] = None
        return result_base
        return result_base
    return wrapper

@benchmark_task
def ner_spacy(text: str, nlp: spacy.language.Language) -> Dict[str, Any]:
    """
    Extract named entities of type "PERSON" from the given text using a spaCy language model.

    Args:
        text (str): The input text to process for named entity recognition.
        nlp (spacy.language.Language): A spaCy Language object representing the loaded spaCy model.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "entities" (List[Tuple[str, str]]): A list of tuples where each tuple contains the entity text 
              and its label (only entities with the label "PERSON" are included).
            - "unique_characters" (List[str]): A sorted list of unique entity texts (up to TOP_N_EXAMPLES).
            - "count" (int): The total count of unique entity texts.

    Raises:
        RuntimeError: If the provided spaCy model (`nlp`) is None.
        TypeError: If the provided `nlp` is not an instance of `spacy.language.Language`.

    Notes:
        - The function ensures that only the 'ner' pipeline is enabled during processing.
        - If the input text is empty, the function returns default values with no entities.
    """
    if nlp is None: raise RuntimeError("spaCy model (nlp_lg) is None in ner_spacy")
    if not isinstance(nlp, spacy.language.Language): raise TypeError("Input 'nlp' must be a spaCy Language object.")
    if not text: return {"entities": [], "unique_characters": [], "count": 0}
    # Ensure necessary pipes are enabled, especially if disabled globally/previously
    with nlp.select_pipes(enable=["ner"]):  # Ensure only the 'ner' pipeline is enabled
        doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "PERSON"]
    unique_characters = sorted(list(set(ent[0] for ent in entities)))
    return {"entities": entities, "unique_characters": unique_characters[:TOP_N_EXAMPLES], "count": len(unique_characters)}

@benchmark_task
def ner_nltk(text: str) -> Dict[str, Any]:
    """
    Perform Named Entity Recognition (NER) on the input text using NLTK.
    This function identifies named entities of type "PERSON" in the given text
    and returns a dictionary containing the extracted entities, unique character
    names, and their count. If required NLTK resources are not available, they
    will be downloaded automatically.
    Args:
        text (str): The input text to process for named entity recognition.
    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - "entities" (List[Tuple[str, str]]): A list of tuples where each tuple
              contains a named entity and its type (e.g., ("John Doe", "PERSON")).
            - "unique_characters" (List[str]): A sorted list of unique character names
              extracted from the entities.
            - "count" (int): The count of unique character names.
    Notes:
        - If the input text is empty, the function returns an empty result with
          default values for "entities", "unique_characters", and "count".
        - The function ensures that the required NLTK resources are available
          before processing the text.
    """
    # Ensure required NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('chunkers/maxent_ne_chunker')
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

    if not text: return {"entities": [], "unique_characters": [], "count": 0}
    entities = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        tokens = nltk.word_tokenize(sent); tagged = nltk.pos_tag(tokens); chunks = nltk.ne_chunk(tagged)
        for chunk in chunks:
            if hasattr(chunk, "label") and chunk.label() == "PERSON":
                name = " ".join(c[0] for c in chunk.leaves()); entities.append((name, "PERSON"))
    unique_characters = sorted(list(set(entity[0] for entity in entities)))
    return {"entities": entities, "unique_characters": unique_characters[:TOP_N_EXAMPLES], "count": len(unique_characters)}

@benchmark_task
def sentiment_spacy(text: str, nlp: spacy.language.Language) -> Dict[str, Any]:
    """
    Analyze the sentiment of a given text using a spaCy language model with the spacytextblob pipeline.

    This function identifies emotionally charged sentences in the input text based on polarity and subjectivity
    thresholds. It returns the most emotional sentences along with their polarity scores.

    Args:
        text (str): The input text to analyze.
        nlp (spacy.language.Language): A spaCy language model instance. The model must support the spacytextblob
            pipeline. If the pipeline is not already added, it will be added automatically.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "emotional_sentences" (List[Tuple[str, float]]): A list of tuples where each tuple contains an
            emotionally charged sentence and its rounded polarity score. The list is sorted by the absolute
            value of polarity in descending order, and only the top N examples are included.
            - "count" (int): The total number of emotionally charged sentences identified.

    Raises:
        RuntimeError: If the provided spaCy model (`nlp`) is None.
        TypeError: If the provided `nlp` is not an instance of `spacy.language.Language`.

    Notes:
        - The function requires the `spacytextblob` pipeline to be available in the spaCy model.
        - The `parser` component must be enabled in the spaCy pipeline to process sentence boundaries (`doc.sents`).
        - The thresholds for polarity and subjectivity, as well as the number of top examples to return, are
        determined by the constants `SENTIMENT_POLARITY_THRESHOLD`, `SENTIMENT_SUBJECTIVITY_THRESHOLD`, and
        `TOP_N_EXAMPLES`, which must be defined elsewhere in the code.
    """
    if nlp is None: raise RuntimeError("spaCy model (nlp_lg) is None in sentiment_spacy")
    if not isinstance(nlp, spacy.language.Language): raise TypeError("Input 'nlp' must be a spaCy Language object.")
    if "spacytextblob" not in nlp.pipe_names:
        nlp.add_pipe("spacytextblob")
    if not text: return {"emotional_sentences": [], "count": 0}
    with nlp.select_pipes(enable=["tagger", "attribute_ruler", "parser", "spacytextblob"]):  # Ensure needed pipes
        doc = nlp(text)
    emotional_sentences = []
    for sent in doc.sents:  # Ensure `doc.sents` is enabled by the `parser` component
        polarity = sent._.blob.polarity; subjectivity = sent._.blob.subjectivity
        if abs(polarity) >= SENTIMENT_POLARITY_THRESHOLD and subjectivity >= SENTIMENT_SUBJECTIVITY_THRESHOLD:
            emotional_sentences.append((sent.text.strip(), round(polarity, 3)))
    emotional_sentences.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"emotional_sentences": emotional_sentences[:TOP_N_EXAMPLES], "count": len(emotional_sentences)}

@benchmark_task
def sentiment_nltk(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of a given text using NLTK's SentimentIntensityAnalyzer.

    This function tokenizes the input text into sentences, calculates the sentiment
    scores for each sentence, and returns the most emotionally charged sentences
    along with their sentiment scores.

    Args:
        text (str): The input text to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "emotional_sentences" (List[Tuple[str, float]]): A list of tuples where each
            tuple contains a sentence and its corresponding sentiment score, sorted
            by the absolute value of the sentiment score in descending order. Only
            the top `TOP_N_EXAMPLES` sentences are included.
            - "count" (int): The total number of sentences analyzed.

    Raises:
        RuntimeError: If the SentimentIntensityAnalyzer fails to initialize.

    Notes:
        - The `TOP_N_EXAMPLES` variable must be defined globally to determine the
        number of top emotional sentences to return.
        - The `sentiment_analyzer` variable is expected to be a global instance of
        SentimentIntensityAnalyzer.
    """
    global sentiment_analyzer
    if sentiment_analyzer is None:
        try:
            sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NLTK SentimentIntensityAnalyzer: {e}")
    if not text: return {"emotional_sentences": [], "count": 0}
    sid = sentiment_analyzer; sentences = nltk.sent_tokenize(text); sentence_sentiments = []
    for sentence in sentences:
        scores = sid.polarity_scores(sentence); compound_score = scores['compound']
        sentence_sentiments.append((sentence.strip(), round(compound_score, 3)))
    sentence_sentiments.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"emotional_sentences": sentence_sentiments[:TOP_N_EXAMPLES], "count": len(sentence_sentiments)}

@benchmark_task
def speed_spacy(text: str, nlp: spacy.language.Language) -> Dict[str, Any]:
    """
    Measure the processing speed of a defined spaCy pipeline on a given text
    by running multiple iterations.

    Args:
        text (str): The input text to be processed.
        nlp (spacy.language.Language): The loaded spaCy language model.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "text_length" (int): The length of the input text.
            - "iterations" (int): The number of iterations performed.
            # 'time_taken' (for all iterations) added by the decorator
    """
    if nlp is None:
        raise RuntimeError("spaCy model (nlp_lg) is None in speed_spacy")
    if not isinstance(nlp, spacy.language.Language):
        raise TypeError("Input 'nlp' must be a spaCy Language object.")
    if not isinstance(text, str):
        raise TypeError(f"Input 'text' must be a string, got {type(text)}")
    if not text:
        return {"text_length": 0, "iterations": 0}

    # --- INCREASED ITERATIONS ---
    # Increase significantly more to ensure measurable time
    # Start with 500, adjust higher/lower based on results
    iterations = 500 # Adjust as needed (e.g., 200, 500, 1000)

    # Define the specific pipeline components to enable for this speed test
    # Ensure this list represents the workload you want to measure
    components_to_enable = ["tok2vec", "tagger", "ner"]

    # Check if required components exist in the model
    for component in components_to_enable:
        if component not in nlp.pipe_names:
            raise ValueError(f"Required component '{component}' not found in spaCy pipeline.")

    # Process the text multiple times within the timed block
    with nlp.select_pipes(enable=components_to_enable):
        for _ in range(iterations): # Use the increased iterations count
            doc = nlp(text)
            # Ensure some work is done by accessing attributes
            _ = [token.tag_ for token in doc if hasattr(token, "tag_")]
            _ = [ent.label_ for ent in doc.ents if hasattr(ent, "label_")]

    # Return info needed for CPS calculation in analysis step
    return {
        "text_length": len(text),
        "iterations": iterations
    }

@benchmark_task
def speed_nltk(text: str) -> Dict[str, Any]:
    """
    Measure the processing speed of NLTK on a given text by running
    multiple iterations.

    Args:
        text (str): The input text to be processed.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "text_length" (int): The length of the input text.
            - "iterations" (int): The number of iterations performed.
            # 'time_taken' (for all iterations) added by the decorator
    """
    if not text:
        return {"text_length": 0, "iterations": 0}

    # Ensure required NLTK resources are available
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    for resource_path, package in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                # Consider quiet=False during debugging if downloads fail
                nltk.download(package, quiet=True)
            except Exception as e:
                raise RuntimeError(f"Failed to download required NLTK resource {package}: {e}")

    # --- INCREASED ITERATIONS ---
    # Increase significantly more to ensure measurable time
    # Match spaCy's count if desired for direct comparison, or adjust independently
    iterations = 500 # Adjust as needed (e.g., 200, 500, 1000)

    # Process text multiple times to get stable performance measure
    for _ in range(iterations): # Use the increased iterations count
        # Simulate a typical NLTK workload
        sentences = nltk.sent_tokenize(text)
        tokens = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tags = [nltk.pos_tag(sent_tokens) for sent_tokens in tokens]
        # NER chunking (can be slow, ensures workload)
        _ = [nltk.ne_chunk(sent_tags) for sent_tags in pos_tags]
        # Simulate accessing tokens (like lemmatization)
        _ = [token.lower() for sent_tokens in tokens for token in sent_tokens]

    # Return info needed for CPS calculation in analysis step
    return {
        "text_length": len(text),
        "iterations": iterations
    }
    
@benchmark_task
def dialogue_detection_spacy(text: str, nlp: spacy.language.Language) -> Dict[str, Any]:
    """
    Detect dialogue snippets and attempt speaker attribution using spaCy.

    Approach:
    1. Finds text within double ("") or single ('') quotes within sentences.
    2. Uses spaCy's Matcher to find potential dialogue tags (PERSON + VERB).
    3. Attempts to link quotes to speakers identified via NER near dialogue verbs
        using dependency parse information (`nsubj`).
    4. Returns snippets as (speaker, dialogue_text) tuples.

    Limitations:
    - Primarily handles quotes contained within single sentences.
    - Speaker attribution is heuristic and may fail with implicit speakers
        or complex sentence structures.
    - May misinterpret quotes used for emphasis, titles, or nested quotes.
    - Does not robustly distinguish internal monologue.
    """
    if nlp is None:
        raise RuntimeError("spaCy model (nlp_lg) is None in dialogue_detection_spacy. Ensure the model is loaded using spacy.load(SPACY_MODEL).")
    if not isinstance(nlp, spacy.language.Language): raise TypeError("Input 'nlp' must be a spaCy Language object.")
    if not text: return {"dialogue_snippets": [], "count": 0}
    # Ensure parser and ner components are enabled for dependency and entity information
    if "parser" not in nlp.pipe_names or "ner" not in nlp.pipe_names:
        raise RuntimeError("spaCy pipeline is missing required components: 'parser' and 'ner'. Ensure they are enabled.")
    with nlp.select_pipes(enable=["parser", "ner"]):  # Tok2Vec/Transformer needed implicitly
        doc = nlp(text)

    dialogue_snippets = []

    # Regex to find potential quoted segments (non-greedy)
    # Handles simple double and single quotes. More complex regex needed for escaped quotes etc.
    quote_pattern = r'([\'"])(.*?)\1'

    for sent in doc.sents:
        # Find all potential quotes in the sentence
        found_quotes = re.findall(quote_pattern, sent.text)

        if not found_quotes:
            continue # Skip sentence if no quotes found

        # --- Attempt Speaker Attribution ---
        speaker = None
        # Removed unused variable `dialogue_verb_token`

        # Look for dialogue verbs and check their subjects
        for token in sent:
            if token.lemma_ in DIALOGUE_VERBS and token.pos_ == 'VERB':
                # Found a potential dialogue verb (removed unused variable `dialogue_verb_token`)
                # Check subject of the verb using dependency parse
                for child in token.children:
                    if child.dep_ == "nsubj" and child.ent_type_ == "PERSON":  # Dependency and entity type checks
                        speaker = child.text
                        break # Found speaker via direct subject
                if speaker: break # Found speaker for this verb, move on

                # If no direct subject, check ancestors? (More complex, might skip)
                # head = token
                # while head.head != head and not speaker:
                #     if head.dep_ == "nsubj" and head.ent_type_ == "PERSON":  # Dependency and entity type checks
                #         speaker = head.text
                #     head = head.head
                # if speaker: break


        # If no speaker found via verb subject, check if there's only one PERSON in the sentence
        if not speaker:
            persons_in_sent = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]  # Extract PERSON entities
            if len(persons_in_sent) == 1:
                # If only one person mentioned, tentatively assign them as speaker
                # This is a heuristic and might be wrong
                speaker = persons_in_sent[0]
            # elif len(persons_in_sent) > 1:
                # speaker = "Ambiguous" # Or leave as None


        # Add all found quotes from this sentence with the determined speaker
        for _, quote_text in found_quotes:  # Removed unused `_quote_mark`
            quote_text = quote_text.strip()
            # Basic filter: avoid very short quotes or quotes that look like titles/emphasis
            if len(quote_text) > 2 and not quote_text.istitle():
                dialogue_snippets.append((speaker, quote_text)) # Store as (Speaker, Text)

    # Post-processing idea: If consecutive snippets have the same speaker, could merge them? (Complex)

    return {
        # Return tuples: (speaker_name or None, dialogue_text)
        "dialogue_snippets": dialogue_snippets[:TOP_N_EXAMPLES],
        "count": len(dialogue_snippets),
    }

@benchmark_task
def dialogue_detection_nltk(text: str) -> Dict[str, Any]:
    """
    Detect dialogue snippets and attempt speaker attribution using NLTK.

    Approach:
    1. Finds text within double ("") or single ('') quotes within sentences.
    2. Uses NLTK's POS tagging and NER to find potential speakers (PERSON)
       and dialogue verbs.
    3. Attempts to link quotes to speakers if a PERSON and dialogue VERB
       are found within the same sentence.
    4. Returns snippets as (speaker, dialogue_text) tuples.

    Limitations:
    - Primarily handles quotes contained within single sentences.
    - Speaker attribution is basic (checks for co-occurrence in sentence)
      and may fail often or be ambiguous. NLTK's NER can be less accurate.
    - May misinterpret quotes used for emphasis, titles, or nested quotes.
    - Does not robustly distinguish internal monologue.
    """
    if not text: return {"dialogue_snippets": [], "count": 0}

    dialogue_snippets = []
    sentences = nltk.sent_tokenize(text)

    # Regex to find potential quoted segments (non-greedy)
    quote_pattern = r'([\'"])(.*?)\1'

    for sent_text in sentences:
        # Find all potential quotes in the sentence
        found_quotes = re.findall(quote_pattern, sent_text)

        if not found_quotes:
            continue # Skip sentence if no quotes found

        # --- Attempt Speaker Attribution (Basic) ---
        speaker = None
        try:
            tokens = nltk.word_tokenize(sent_text)
            tagged_tokens = nltk.pos_tag(tokens)

            # Check for dialogue verbs
            has_dialogue_verb = any(word.lower() in DIALOGUE_VERBS_NLTK and tag.startswith('VB')
                                   for word, tag in tagged_tokens)

            # Check for PERSON entities using NLTK's NER chunking
            persons_in_sent = []
            # Only run NER if needed (optimization - if no quotes, skip NER too?)
            if has_dialogue_verb: # Only look for speakers if a dialogue verb exists
                 tree = nltk.ne_chunk(tagged_tokens)
                 for chunk in tree.subtrees(filter=lambda t: t.label() == 'PERSON'):
                      persons_in_sent.append(" ".join(c[0] for c in chunk.leaves()))

            # Heuristic: If exactly one PERSON and at least one dialogue VERB, assign speaker
            if len(persons_in_sent) == 1 and has_dialogue_verb:
                speaker = persons_in_sent[0]
            # elif len(persons_in_sent) > 1 and has_dialogue_verb:
            #      speaker = "Ambiguous" # Or leave as None

        except Exception as e:
             # Log potential NLTK errors during tagging/chunking
             logger = logging.getLogger("nlp_benchmark")
             logger.warning(f"NLTK processing error during dialogue speaker check for sentence '{sent_text[:50]}...': {e}")
             speaker = None # Default to no speaker on error


        # Add all found quotes from this sentence with the determined speaker
        for _quote_mark, quote_text in found_quotes:
            quote_text = quote_text.strip()
            # Basic filter: avoid very short quotes or quotes that look like titles/emphasis
            if len(quote_text) > 2 and not quote_text.istitle():
                 dialogue_snippets.append((speaker, quote_text))

    return {
        # Return tuples: (speaker_name or None, dialogue_text)
        "dialogue_snippets": dialogue_snippets[:TOP_N_EXAMPLES],
        "count": len(dialogue_snippets),
    }

def benchmark_performance(text: str, task: str = "all", libraries: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Benchmarks the performance of various NLP tasks using specified libraries.

    Args:
        text (str): The input text to process.
        task (str, optional): The specific NLP task to benchmark. Options are:
            - "all": Run all tasks (default).
            - "minimal": Run only the "processing_speed" task.
            - "ner": Named Entity Recognition.
            - "sentiment": Sentiment analysis.
            - "dialogue_detection": Dialogue detection.
            - "processing_speed": Measure processing speed.
        libraries (Optional[List[str]], optional): List of libraries to use for benchmarking.
            Defaults to ["spacy", "nltk"] if not provided.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: A nested dictionary containing the results of the benchmark.
            The structure is as follows:
            {
                task_name: {
                    library_name: {
                        "time_taken": float,  # Time taken to execute the task (if applicable).
                        "chars_per_second": float,  # Characters processed per second (for processing_speed task).
                        "error": str,  # Error message if the task failed.
                        ...  # Additional task-specific results.
                    }
                }
            }

    Notes:
        - If the "spacy" library is used, the spaCy model (nlp_lg) is initialized if not already loaded.
        - If the "sentiment" task is run with "nltk", the NLTK Sentiment Analyzer is initialized if not already available.
        - Tasks that fail due to missing dependencies or errors will include an "error" key in their results.
        - The "processing_speed" task calculates the number of characters processed per second.

    Raises:
        Exception: If critical errors occur during task execution, they are logged, and the error details are included in the results.
    """
    global nlp_lg, sentiment_analyzer # Add sentiment_analyzer here
    task_logger = logging.getLogger("nlp_benchmark")
    if libraries is None: libraries = ["spacy", "nltk"]
    if nlp_lg is None and "spacy" in libraries:
        try:
            task_logger.info("Initializing spaCy model (nlp_lg)...")
            nlp_lg = spacy.load(SPACY_MODEL)
            if "spacytextblob" not in nlp_lg.pipe_names:
                nlp_lg.add_pipe("spacytextblob")
            task_logger.info("spaCy model initialized successfully.")
        except Exception as e:
            task_logger.warning(f"spaCy model initialization failed: {format_error(e)}. Skipping spaCy tasks.")
            libraries = [lib for lib in libraries if lib != "spacy"]
            if not libraries:
                return {}
    tasks_to_run: List[str] = []
    if task == "all": tasks_to_run = ["ner", "sentiment", "dialogue_detection", "processing_speed"]
    elif task == "minimal": tasks_to_run = ["processing_speed"]
    elif task in ["ner", "sentiment", "dialogue_detection", "processing_speed"]: tasks_to_run = [task]
    else: task_logger.warning(f"Unknown task: {task}. Running 'all'."); tasks_to_run = ["ner", "sentiment", "dialogue_detection", "processing_speed"]
    results: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for t in tasks_to_run:
        for lib in libraries:
            if t == "sentiment" and sentiment_analyzer is None:
                try:
                    task_logger.info("Initializing NLTK Sentiment Analyzer...")
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    task_logger.info("NLTK Sentiment Analyzer initialized successfully.")
                except Exception as e:
                    task_logger.error(f"Failed to initialize NLTK Sentiment Analyzer: {format_error(e)}")
                    lib_result = {"error": "NLTK Sentiment Analyzer unavailable"}
            try:
                func_name = f"{t}_{lib}"; task_func = globals().get(func_name)
                if task_func is None: lib_result = {"error": f"Task function {func_name} not found"}
                else:
                    if lib == "spacy":
                        if nlp_lg is None: lib_result = {"error": "spaCy model unavailable during task execution"}
                        else: args = [text, nlp_lg]; lib_result = task_func(*args)
                    elif lib == "nltk":
                        if t == "sentiment" and sentiment_analyzer is None: lib_result = {"error": "NLTK Sentiment Analyzer unavailable"}
                        else: args = [text]; lib_result = task_func(*args)
                if t == "processing_speed" and lib_result.get("time_taken", 0) > 0: lib_result["chars_per_second"] = round(lib_result.get("text_length", 0) / lib_result["time_taken"], 2)
                elif t == "processing_speed": lib_result["chars_per_second"] = 0.0
                results[t][lib] = lib_result
            except Exception as e: error_msg = f"Critical error calling {t}_{lib}: {format_error(e)}"; task_logger.error(error_msg, exc_info=True); results[t][lib] = {"time_taken": 0.0, "error": error_msg}
    return dict(results)

def _process_chunk_or_sample(sample_id: str, text: str, chunk_id: Optional[str] = None, chunk_num: Optional[int] = None, total_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Processes a text sample or chunk, benchmarks its performance using various NLP libraries, 
    and returns the results in a structured format.

    Args:
        sample_id (str): Unique identifier for the sample being processed.
        text (str): The text content of the sample or chunk to be processed.
        chunk_id (Optional[str], optional): Identifier for the chunk, if applicable. Defaults to None.
        chunk_num (Optional[int], optional): The position of the chunk in the sequence of chunks. Defaults to None.
        total_chunks (Optional[int], optional): Total number of chunks in the sample. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the processing results. Each dictionary includes:
            - sample_id (str): The identifier of the sample.
            - sample_length (int): The length of the text sample.
            - task (str): The NLP task performed (e.g., "dialogue_detection", "ner", etc.).
            - library (str): The NLP library used for the task.
            - time_taken (float): Time taken to perform the task.
            - error (Optional[str]): Error message, if any occurred during processing.
            - merged (bool): Indicates if the result is merged.
            - chunk_id (Optional[str]): Identifier for the chunk, if applicable.
            - chunk_num (Optional[int]): Position of the chunk in the sequence, if applicable.
            - total_chunks (Optional[int]): Total number of chunks, if applicable.
            - chunk_size (Optional[int]): Size of the chunk, if applicable.
            - chunk_position (str): Position of the chunk in the sequence ("first", "middle", "last", "only", or "full_sample").
            - Additional task-specific fields:
                - For "dialogue_detection": "count" (int), "sample_results" (list of dialogue sentences).
                - For "ner": "count" (int), "sample_results" (list of unique characters).
                - For "sentiment": "count" (int), "sample_results" (list of emotional sentences).
                - For "processing_speed": "chars_per_second" (float), "text_length" (int).

    Raises:
        MemoryError: If a memory error occurs during processing, a fallback mechanism is triggered.
        Exception: If any other exception occurs, it is logged and returned as part of the result.

    Notes:
        - The function uses the `benchmark_performance` utility to evaluate the text with different NLP tasks and libraries.
        - If a memory error occurs, the function attempts a fallback with reduced tasks and libraries.
        - Logging is used extensively to capture errors, warnings, and debug information.
    """
    proc_logger = logging.getLogger("nlp_benchmark")
    if not isinstance(sample_id, str) or not sample_id: proc_logger.error("Invalid sample_id"); return [{"error": "Invalid sample_id", "sample_id": sample_id, "chunk_id": chunk_id}]
    if not isinstance(text, str): proc_logger.warning(f"Non-string text for {sample_id}, chunk {chunk_id}."); return [{"error": "Non-string text", "sample_id": sample_id, "chunk_id": chunk_id}]
    proc_logger.debug(f"Processing {sample_id}, chunk {chunk_id or 'N/A'} (Length: {len(text)})")
    processed_results = []; text_length = len(text); run_task = "all"
    benchmark_results = {}
    try: benchmark_results = benchmark_performance(text=text, task=run_task, libraries=["spacy", "nltk"])
    except MemoryError:
        proc_logger.warning(f"Memory error {sample_id=}, {chunk_id=}. Falling back."); clear_memory(); run_task = "minimal"
        try: benchmark_results = benchmark_performance(text=text, task=run_task, libraries=["nltk"])
        except Exception as fallback_e: error_msg = f"Fallback failed: {format_error(fallback_e)}"; proc_logger.error(error_msg, exc_info=True); return [{"sample_id": sample_id, "chunk_id": chunk_id, "error": error_msg, "task":"processing_error"}]
    except Exception as e: error_msg = f"Error in benchmark_performance {sample_id=}, {chunk_id=}: {format_error(e)}"; proc_logger.error(error_msg, exc_info=True); return [{"sample_id": sample_id, "chunk_id": chunk_id, "error": error_msg, "task":"processing_error"}]
    for task, lib_results in benchmark_results.items():
        for lib, res in lib_results.items():
            if not isinstance(res, dict): proc_logger.error(f"Result for {task}/{lib} not dict: {res}."); res = {"error": "Invalid result format"}
            result_row = {"sample_id": sample_id, "sample_length": text_length, "task": task, "library": lib, "time_taken": res.get("time_taken", 0.0), "error": res.get("error"), "merged": False}
            if chunk_id and chunk_num is not None and total_chunks is not None:
                result_row.update({"chunk_id": chunk_id, "chunk_num": chunk_num, "total_chunks": total_chunks, "chunk_size": text_length})
                if chunk_num == 1 and total_chunks == 1: result_row["chunk_position"] = "only"
                elif chunk_num == 1: result_row["chunk_position"] = "first"
                elif chunk_num == total_chunks: result_row["chunk_position"] = "last"
                else: result_row["chunk_position"] = "middle"
            else: result_row.update({"chunk_id": None, "chunk_num": None, "total_chunks": None, "chunk_size": None, "chunk_position": "full_sample"})
            if task == "dialogue_detection":
                # Store the list of (speaker, text) tuples
                result_row["sample_results"] = res.get("dialogue_snippets")
                result_row["count"] = res.get("count")
            elif task == "ner": result_row.update({"count": res.get("count"), "sample_results": res.get("unique_characters")})
            elif task == "sentiment": result_row.update({"count": res.get("count"), "sample_results": res.get("emotional_sentences")})
            elif task == "processing_speed": result_row.update({"chars_per_second": res.get("chars_per_second"), "text_length": res.get("text_length")})
            processed_results.append(result_row)
    return processed_results

def merge_chunk_results(chunk_results: List[Dict[str, Any]], original_sample_length: int) -> List[Dict[str, Any]]:
    """
    Merges the results of processing chunks of data into a consolidated format.

    Args:
        chunk_results (List[Dict[str, Any]]): A list of dictionaries containing the results of processing individual chunks.
            Each dictionary should include keys such as "task", "library", "chunk_num", "total_chunks", "error", "time_taken",
            "count", and "sample_results".
        original_sample_length (int): The length of the original sample being processed.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the merged results. Each dictionary contains information
            about the task, library, processing time, errors (if any), and aggregated results. If no valid results are
            available, the function returns error results.

    Notes:
        - The function groups chunk results by task and library, sorts them by chunk number, and aggregates relevant data.
        - For tasks like "ner", "dialogue_detection", and "sentiment", it aggregates unique sample results and counts them.
        - For "processing_speed", it calculates the characters processed per second.
        - Errors encountered during chunk processing are logged and included in the merged results.
        - The function handles edge cases such as missing or invalid data gracefully, logging warnings where appropriate.

    Raises:
        None: The function does not raise exceptions but logs warnings for unexpected data or processing issues.
    """
    if not chunk_results: return []
    merge_logger = logging.getLogger("nlp_benchmark")
    error_results = [r for r in chunk_results if r.get("error") and r.get("task") == "processing_error"]
    valid_results = [r for r in chunk_results if not (r.get("error") and r.get("task") == "processing_error")]
    if not valid_results: 
        sample_id = chunk_results[0].get('sample_id', 'Unknown') if chunk_results else 'Unknown'
        merge_logger.warning(f"No valid chunk results to merge for {sample_id}")
        return error_results
    grouped = defaultdict(list); sample_id = valid_results[0]["sample_id"]
    for res in valid_results: key = (res["task"], res["library"]); grouped[key].append(res)
    merged_results: List[Dict[str, Any]] = []
    for (task, library), group_results in grouped.items():
        group_results.sort(key=lambda x: x.get("chunk_num", 0))
        merged_record: Dict[str, Any] = {"sample_id": sample_id, "sample_length": original_sample_length, "task": task, "library": library, "merged": True, "chunks_processed": len(group_results), "total_chunks": group_results[0].get("total_chunks"), "error": None}
        total_time = sum(r.get("time_taken", 0.0) for r in group_results); merged_record["time_taken"] = total_time
        if task == "processing_speed": merged_record["chars_per_second"] = round(original_sample_length / total_time, 2) if total_time > 0 else 0.0; merged_record["text_length"] = original_sample_length
        total_count = 0; aggregated_samples: List[Any] = []; seen_samples = set(); chunk_errors = []
        for r in group_results:
            if r.get("error"): chunk_errors.append(f"Chunk {r.get('chunk_num', '?')}: {r['error']}")
            if not r.get("error"):
                 current_count = r.get("count")
                 if isinstance(current_count, (int, float)): total_count += current_count
                 samples = r.get("sample_results")
                 if samples and isinstance(samples, list):
                     for sample_item in samples:
                         try:
                              hashable_item = tuple(sample_item) if isinstance(sample_item, list) else sample_item
                              if isinstance(hashable_item, (str, int, float, tuple, bool)):
                                   if hashable_item not in seen_samples: aggregated_samples.append(sample_item); seen_samples.add(hashable_item)
                         except TypeError: merge_logger.warning(f"Could not hash sample item: {sample_item}", exc_info=False)
        if chunk_errors: merged_record["error"] = "; ".join(chunk_errors)
        if task in ["ner", "dialogue_detection", "sentiment"]:
             merged_record["count"] = len(aggregated_samples)
             if task == "sentiment" and aggregated_samples:
                  try: aggregated_samples.sort(key=lambda x: abs(x[1]) if isinstance(x, (list, tuple)) and len(x) > 1 and isinstance(x[1], (int, float)) else 0, reverse=True)
                  except (TypeError, IndexError) as sort_err: merge_logger.warning(f"Could not re-sort merged sentiment: {sort_err}")
             if task == "dialogue_detection":
                merged_record["count"] = len(aggregated_samples) # Deduplicated count
                # aggregated_samples now contains (speaker, text) tuples
                merged_record["sample_results"] = aggregated_samples[:TOP_N_EXAMPLES]
        elif task == "processing_speed": merged_record["count"] = None; merged_record["sample_results"] = None
        else: merged_record["count"] = total_count if isinstance(total_count, (int, float)) else None; merged_record["sample_results"] = None
        merged_results.append(merged_record)
    merged_results.extend(error_results); return merged_results

def process_sample(sample_id: str, sample_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Dict[str, Any]]:
    """
    Processes a text sample by splitting it into chunks, processing each chunk,
    and merging the results.

    Args:
        sample_id (str): A unique identifier for the sample. Must be a non-empty string.
        sample_text (str): The text content of the sample to be processed.
        chunk_size (int, optional): The size of each chunk in characters. Must be a positive integer.
            Defaults to the global constant CHUNK_SIZE.
        overlap (int, optional): The number of overlapping characters between consecutive chunks.
            Must be a non-negative integer and smaller than `chunk_size`.
            Defaults to the global constant OVERLAP.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the processed results.
            If an error occurs during processing or merging, the list may contain
            an error dictionary with details about the issue. Returns an empty list
            for empty input samples.

    Raises:
        ValueError: If `sample_id` is empty, `chunk_size` is not positive, or `overlap` is
            negative or greater than or equal to `chunk_size`.
        TypeError: If `sample_id` or `sample_text` are not strings, or `chunk_size` or `overlap`
            are not integers.

    Notes:
        - If the sample text is shorter than or equal to the chunk size, it is processed as a
            single unit without splitting.
        - Logs information, warnings, and errors using the logger named "nlp_benchmark".
        - Handles exceptions gracefully, returning error details in the result list.
        - Ensures memory cleanup by calling `clear_memory()` in the `finally` block.
    """
    ps_logger = logging.getLogger("nlp_benchmark")
    all_chunk_results: List[Dict[str, Any]] = [] # Initialize list to hold chunk results

    try:
        # --- Input Validation ---
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string.")
        if not isinstance(sample_text, str):
            raise TypeError("sample_text must be a string.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        original_length = len(sample_text)
        ps_logger.info(f"Processing sample: {sample_id} (length: {original_length}, chunk_size: {chunk_size}, overlap: {overlap})")

        # --- Handle Empty Sample ---
        if not sample_text:
            ps_logger.warning(f"Sample {sample_id} is empty. Returning empty list.")
            return []

        # --- Handle Small Sample (No Chunking Needed) ---
        if original_length <= chunk_size:
            ps_logger.info(f"Processing {sample_id} as a single unit (length <= chunk_size).")
            # Process the entire text as one chunk
            single_result = _process_chunk_or_sample(
                sample_id=sample_id,
                text=sample_text,
                chunk_id=f"{sample_id}#full", # Indicate it's the full sample
                chunk_num=1,
                total_chunks=1
            )
            # Return the result list, or an error if processing failed
            return single_result if single_result else [{"sample_id": sample_id, "error": "No results from single chunk processing", "task": "processing_error"}]

        # --- Handle Large Sample (Chunking Required) ---
        ps_logger.info(f"Splitting {sample_id} into chunks...")
        chunks: List[str] = []
        chunk_ids: List[str] = []
        step = chunk_size - overlap # Calculate step size for overlap
        chunk_count = 0

        # Generate chunks
        for i in range(0, original_length, step):
            chunk_start = i
            chunk_end = min(i + chunk_size, original_length)
            chunks.append(sample_text[chunk_start:chunk_end])
            chunk_count += 1
            chunk_ids.append(f"{sample_id}#chunk{chunk_count}")
            if chunk_end == original_length:
                break # Stop if we've reached the end

        total_chunks_generated = len(chunks)
        ps_logger.info(f"Generated {total_chunks_generated} chunks for {sample_id}. Processing each chunk...")

        # Process each chunk sequentially (parallelism happens in process_multiple_samples)
        for i, (chunk_text, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            chunk_num = i + 1
            ps_logger.debug(f"Processing chunk {chunk_num}/{total_chunks_generated} for {sample_id} (ID: {chunk_id})")
            chunk_results = _process_chunk_or_sample(
                sample_id=sample_id,
                text=chunk_text,
                chunk_id=chunk_id,
                chunk_num=chunk_num,
                total_chunks=total_chunks_generated
            )
            if chunk_results: # Ensure results were returned
                all_chunk_results.extend(chunk_results)
            else:
                ps_logger.warning(f"No results returned for chunk {chunk_id}. Appending error placeholder.")
                all_chunk_results.append({
                    "sample_id": sample_id,
                    "chunk_id": chunk_id,
                    "chunk_num": chunk_num,
                    "total_chunks": total_chunks_generated,
                    "error": f"Processing function returned no results for chunk {chunk_num}",
                    "task": "processing_error"
                })

        # --- Merge Chunk Results ---
        ps_logger.info(f"Finished processing all {total_chunks_generated} chunks for {sample_id}. Merging results...")
        merged_sample_results = merge_chunk_results(all_chunk_results, original_length)

        if not merged_sample_results:
            # This might happen if merging fails or all chunks had errors
            ps_logger.error(f"Failed to merge chunks or no valid chunk results for {sample_id}.")
            # Return a specific error indicating merge failure, potentially including chunk errors if available
            # For simplicity, return a single error entry here.
            return [{"sample_id": sample_id, "error": "Failed to merge chunk results or no valid data", "task": "processing_error"}]

        ps_logger.info(f"Successfully merged results for {sample_id}.")
        return merged_sample_results

    except (ValueError, TypeError) as e:
        # Catch configuration/input errors
        error_msg = format_error(e)
        ps_logger.error(f"Configuration or input error for {sample_id}: {error_msg}", exc_info=True)
        # Return a list containing a single error dictionary
        return [{"sample_id": sample_id, "error": error_msg, "task": "processing_error"}]
    except Exception as e:
        # Catch unexpected errors during processing
        error_msg = f"Unhandled exception: {format_error(e)}"
        ps_logger.error(f"Critical error processing {sample_id}: {error_msg}", exc_info=True)
        # Return a list containing a single error dictionary
        return [{"sample_id": sample_id, "error": error_msg, "task": "processing_error"}]
    finally:
        # Ensure memory cleanup attempt happens even if errors occur
        clear_memory()

def load_samples(directory: Path = CORPUS_DIR) -> Dict[str, str]:
    """
    Loads text samples from a specified directory and returns them as a dictionary.

    Args:
        directory (Path): The directory from which the samples will be loaded. Defaults to CORPUS_DIR.

    Returns:
        Dict[str, str]: A dictionary containing sample IDs as keys and the corresponding text content as values.
                        Returns an empty dictionary if the directory is not found or no .txt files are present.

    Logs:
        - Logs an error if the directory is not found.
        - Logs a warning if no .txt files are found in the directory.
        - Logs detailed information about each file loaded successfully.
        - Logs errors for files that fail to load.
    """
    samples: Dict[str, str] = {}
    ls_logger = logging.getLogger("nlp_benchmark")
    ls_logger.info(f"Loading samples from: {directory}")

    if not directory.exists():
        ls_logger.error(f"Directory does not exist: {directory}")
        return {}

    if not directory.is_dir():
        ls_logger.error(f"Path is not a directory: {directory}")
        return {}

    txt_files = list(directory.rglob("*.txt"))
    if not txt_files:
        ls_logger.warning(f"No .txt files found in directory: {directory}")
        return {}

    for filepath in txt_files:
        try:
            sample_id = filepath.relative_to(directory).as_posix()
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                samples[sample_id] = f.read()
            ls_logger.debug(f"Loaded: {sample_id} ({len(samples[sample_id])} chars)")
        except Exception as e:
            ls_logger.error(f"Failed to load {filepath}: {format_error(e)}")
            continue

    ls_logger.info(f"Loaded {len(samples)} samples.")
    return samples

def process_multiple_samples(samples: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Processes multiple text samples in parallel using a process pool executor.

    Args:
        samples (Dict[str, str]): A dictionary where keys are sample IDs (str) and values are sample texts (str).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the results of processing each sample. Each result
        dictionary may include information about the sample ID, processing results, and any errors encountered.

    Notes:
        - Uses the globally defined `init_worker` function for worker initialization.
        - Uses a `ProcessPoolExecutor` for parallel processing.
        - Logging is performed at various stages.
        - Handles exceptions during processing and logs them appropriately.
        - Clears memory after processing is complete.
    """
    pm_logger = logging.getLogger("nlp_benchmark")
    if not isinstance(samples, dict):
        pm_logger.error("Invalid 'samples' input: Must be a dictionary.")
        return []
    if not samples:
        pm_logger.warning("No samples to process.")
        return []

    pm_logger.info(f"Starting parallel processing: {len(samples)} samples, {MAX_WORKERS} workers.")
    pm_logger.info(f"Chunk: {CHUNK_SIZE}, Overlap: {OVERLAP}")
    all_results: List[Dict[str, Any]] = []

    # Optional: Set start method if needed (especially for GPU on Win/macOS)
    # try:
    #     mp.set_start_method('spawn', force=True)
    #     pm_logger.info("Set start method to 'spawn'.")
    # except RuntimeError as e:
    #     pm_logger.warning(f"Could not set start method: {e}")
    # except ValueError as e: # Catches if already set
    #      pm_logger.warning(f"Start method likely already set: {e}")

    # Use the globally defined init_worker for initialization
    # The local init_worker definition has been removed.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        futures = {
            executor.submit(process_sample, sid, sample_text, CHUNK_SIZE, OVERLAP): sid
            for sid, sample_text in samples.items()
        }
        pm_logger.info(f"Submitted {len(futures)} tasks.")

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples", unit="sample"):
            sample_id = futures[future]
            try:
                sample_result: Optional[List[Dict[str, Any]]] = future.result() # Can return None or List
                if sample_result: # Check if list is not empty
                    all_results.extend(sample_result)
                    # Calculate success/error count *within* this sample's results
                    num_success = sum(1 for r in sample_result if not r.get('error') or r.get('task') != 'processing_error')
                    num_error = len(sample_result) - num_success
                    pm_logger.info(f"Completed: {sample_id} ({num_success} results, {num_error} errors within sample processing)")
                elif sample_result is None:
                    pm_logger.warning(f"∅ Completed (future returned None): {sample_id}.")
                else: # Empty list returned
                    pm_logger.warning(f"∅ Completed (returned empty list): {sample_id}.")

            except Exception as e:
                # Log error from the future itself (e.g., worker crashed)
                error_msg = format_error(e)
                pm_logger.error(f"Error retrieving result for {sample_id}: {error_msg}", exc_info=True)
                # Append a specific error record for this sample ID
                all_results.append({
                    "sample_id": sample_id,
                    "error": f"Future failed: {error_msg}",
                    "task": "processing_error", # Assign a task type for errors
                    "library": None,
                    "time_taken": 0.0,
                    # Add other relevant null fields if needed by downstream processing
                })

    pm_logger.info(f"Parallel processing finished. Collected {len(all_results)} result entries total.")
    clear_memory() # Suggest GC after large processing job
    return all_results

def create_ground_truth(sample_id: str, text: str, dialogues: Optional[List[str]] = None, characters: Optional[List[str]] = None, emotions: Optional[List[Tuple[str, float]]] = None) -> bool:
    """
    Creates a ground truth JSON file for a given sample.

    Args:
        sample_id (str): Unique identifier for the sample.
        text (str): Text content of the sample.
        dialogues (Optional[List[str]]): List of dialogues in the sample.
        characters (Optional[List[str]]): List of characters in the sample.
        emotions (Optional[List[Tuple[str, float]]]): List of emotions with scores.

    Returns:
        bool: True if the ground truth file was created successfully, False otherwise.
    """
    gt_logger = logging.getLogger("nlp_benchmark")
    if not isinstance(sample_id, str) or not sample_id:
        gt_logger.error("GT Error: Invalid sample_id.")
        return False
    if not isinstance(text, str):
        gt_logger.error(f"GT Error {sample_id}: text must be a string.")
        return False

    ground_truth = {
        "sample_id": sample_id,
        "text_snippet": text[:500] + "..." if len(text) > 500 else text,
        "dialogues": dialogues or [],
        "characters": characters or [],
        "emotions": emotions or [],
        "last_updated": datetime.datetime.now().isoformat()
    }

    safe_filename = sample_id.replace("/", "_").replace("\\", "_") + ".json"
    filepath = GROUND_TRUTH_DIR / safe_filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=4)
        gt_logger.info(f"Ground truth saved successfully: {sample_id} -> {filepath}")
        return True
    except Exception as e:
        gt_logger.error(f"Failed to save ground truth for {sample_id}: {format_error(e)}")
        return False

def load_ground_truth(sample_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads ground truth data for a given sample ID from a JSON file.
    Args:
        sample_id (str): The identifier for the sample. It must be a non-empty string.
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the ground truth data if the file exists 
        and is successfully loaded, or None if the file does not exist, the sample_id is invalid, 
        or an error occurs during loading.
    Logs:
        - Logs an error if the sample_id is invalid.
        - Logs a debug message if the ground truth file is successfully loaded.
        - Logs an error if there is a JSON decoding error or any other exception during file loading.
        - Logs a debug message if the ground truth file does not exist.
    """
    gt_logger = logging.getLogger("nlp_benchmark")
    if not isinstance(sample_id, str) or not sample_id:
        gt_logger.error("GT Load Error: Invalid sample_id.")
        return None

    safe_filename = sample_id.replace("/", "_").replace("\\", "_") + ".json"
    filepath = GROUND_TRUTH_DIR / safe_filename

    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_logger.debug(f"GT loaded: {sample_id}")
                return gt_data
        except json.JSONDecodeError:
            gt_logger.error(f"GT JSON Error: {filepath}")
            return None
        except Exception as e:
            gt_logger.error(f"GT Load Error {filepath}: {format_error(e)}")
            return None
    else:
        gt_logger.debug(f"No GT file: {filepath}")
        return None
    
def serialize_if_needed(item):
    """
    Safely serialize list/dict items to JSON string for CSV output.
    Handles None, NumPy arrays (by converting to list), lists, and dicts.
    Returns other types unchanged. Avoids ambiguous boolean checks.
    """
    # 1. Handle explicit None first
    if item is None:
        # Using None might be better for CSV consistency than np.nan unless needed
        return None

    # 2. Handle NumPy arrays: Convert to list before further checks
    if isinstance(item, np.ndarray):
        try:
            # Convert numpy array to Python list
            item = item.tolist()
            # After conversion, item could potentially be None if array was empty/scalar NaN?
            if item is None:
                return None
        except Exception as e:
            ana_logger.warning(f"Failed to convert numpy array to list: {e}. Returning raw array string.")
            return str(item) # Return string representation as fallback

    # 3. Check if it's a list or dict (covers lists from array conversion too)
    # This is the primary check for serialization candidates.
    if isinstance(item, (list, dict)):
        try:
            # ensure_ascii=False is good for broader character support
            # allow_nan=False prevents dumping float NaN, which is invalid JSON standard
            # Use default=str as a fallback for non-serializable types within list/dict
            return json.dumps(item, ensure_ascii=False, allow_nan=False, default=str)
        except TypeError as e:
            ana_logger.warning(f"JSON dump failed for item type {type(item)}: {e}. Skipping serialization.")
            # Fallback to string representation if dumping fails
            return str(item)

    # 4. Handle potential scalar float NaN explicitly after other type checks
    # Check if item is a float before calling np.isnan
    if isinstance(item, float) and np.isnan(item):
        return None # Represent NaN as None in CSV

    # 5. Return original item if it wasn't None, array, list, dict, or scalar NaN
    # This covers strings, ints, bools, etc.
    return item

def analyze_and_save_results(all_results: List[Dict[str, Any]],
                            samples: Dict[str, str],
                            results_dir: Path,
                            run_timestamp: str) -> Optional[pd.DataFrame]:
    """
    Analyzes benchmarking results, generates performance summaries, and saves
    the results to CSV and JSON files.

    Args:
        all_results (List[Dict[str, Any]]): A list of dictionaries containing
                                            benchmarking results from processing.
        samples (Dict[str, str]): A dictionary of sample IDs and original text
                                (used for count).
        results_dir (Path): The directory where the results and summary files
                            will be saved.
        run_timestamp (str): A timestamp string used to uniquely identify the
                            results files.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the successfully processed
                                results (excluding errors), or None if analysis fails.

    Raises:
        Exception: Logs and handles any exceptions that occur during analysis
                or saving, returning None on failure.

    Notes:
        - Calculates chars/second correctly for speed tests using iterations.
        - Separates processing errors from successful results.
        - Aggregates performance metrics (mean, median, std).
        - Saves detailed results (including errors) to CSV.
        - Saves run configuration and aggregated summary to JSON.
        - Handles JSON serialization for list/dict columns in CSV output.
    """
    # Logger already defined globally/outside
    if not isinstance(all_results, list):
        ana_logger.error("Analysis input error: all_results is not a list.")
        return None
    if not all_results:
        ana_logger.warning("No results provided to analyze.")
        # Return an empty DataFrame for consistency
        return pd.DataFrame()

    try:
        # --- 1. Initial DataFrame Creation and Type Coercion ---
        results_df = pd.DataFrame(all_results)
        ana_logger.info(f"Created initial DataFrame with {len(results_df)} rows from {len(samples)} samples.")
        ana_logger.debug(f"Available columns: {results_df.columns.tolist()}")

        # Define columns expected to be numeric & coerce them
        numeric_cols = ['time_taken', 'sample_length', 'chunk_size',
                        'count', 'chars_per_second', 'iterations']
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                ana_logger.debug(f"Coerced column '{col}' to numeric.")

        # --- 2. Separate Processing Errors ---
        processing_errors_df = pd.DataFrame() # Initialize empty
        if 'task' in results_df.columns:
            error_mask = results_df['task'] == 'processing_error'
            processing_errors_df = results_df.loc[error_mask].copy()
            results_df = results_df.loc[~error_mask].copy()
            ana_logger.info(f"Separated {len(processing_errors_df)} processing error entries.")
        else:
            ana_logger.warning("'task' column not found, cannot separate errors.")

        if results_df.empty:
            ana_logger.warning("No successful results remain after filtering errors.")
            if not processing_errors_df.empty:
                csv_file = results_dir / f"benchmark_results_{run_timestamp}.csv"
                processing_errors_df.to_csv(csv_file, index=False, encoding='utf-8')
                ana_logger.info(f"Only processing errors found. Saved to: {csv_file}")
            return pd.DataFrame()

        # --- 3. Calculate Correct CPS for Speed Tasks ---
        if 'task' in results_df.columns and 'processing_speed' in results_df['task'].unique():
            speed_mask = results_df['task'] == 'processing_speed'
            # Relaxed check: Only require columns definitely needed and present
            required_cols_exist = ['time_taken', 'text_length']
            if all(col in results_df.columns for col in required_cols_exist):
                # Define a function for safe calculation (handles missing 'iterations')
                def calculate_cps(row):
                    time_taken = row['time_taken']
                    text_length = row['text_length']
                    # Safely get iterations, default to 1 if column missing or value NaN/None
                    iterations = row.get('iterations', 1)
                    # Check iterations type and value robustly
                    if not isinstance(iterations, (int, float)) or pd.isna(iterations) or iterations <= 0:
                        iterations = 1

                    if pd.isna(time_taken) or pd.isna(text_length) or time_taken <= 0:
                        return 0.0
                    return (text_length * iterations) / time_taken

                # Apply the calculation row-wise for speed tasks
                results_df.loc[speed_mask, 'chars_per_second'] = results_df[speed_mask].apply(calculate_cps, axis=1)
                ana_logger.info("Recalculated 'chars_per_second' for processing_speed tasks using iterations.")
            else:
                ana_logger.warning(f"Skipping CPS recalculation: Missing essential columns: {required_cols_exist}")
                # Ensure the column exists but fill potentially missing values with 0 for aggregation
                if 'chars_per_second' not in results_df.columns:
                    results_df['chars_per_second'] = 0.0 # Add column if missing
                results_df.loc[speed_mask, 'chars_per_second'] = results_df.loc[speed_mask, 'chars_per_second'].fillna(0.0)

        # --- 4. Generate Aggregated Summary Statistics ---
        summary = pd.DataFrame()
        agg_cols = ['task', 'library', 'time_taken', 'sample_length', 'chars_per_second', 'sample_id']
        # Check if all necessary columns for aggregation exist *after* potential creation/filling
        if all(col in results_df.columns for col in agg_cols):
            ana_logger.info("\n--- Performance Summary ---")
            try:
                summary = results_df.groupby(['task', 'library']).agg(
                    mean_time=('time_taken', 'mean'),
                    median_time=('time_taken', 'median'),
                    std_time=('time_taken', 'std'),
                    mean_len=('sample_length', 'mean'),
                    mean_cps=('chars_per_second', 'mean'), # Now uses corrected CPS
                    count=('sample_id', 'size')
                ).round(4)
                summary = summary.fillna(0.0) # Fill any remaining NaNs (e.g., std with 1 item)
                ana_logger.info(f"\n{summary.to_string()}")
            except Exception as agg_e:
                ana_logger.error(f"Error during results aggregation: {agg_e}", exc_info=True)
                ana_logger.warning("Skipping summary generation due to aggregation error.")
        else:
            ana_logger.warning(f"Missing one or more required columns for performance summary: {agg_cols}. Skipping summary generation.")

        # --- 5. Log Overall Summary Counts ---
        ana_logger.info("\n===== RESULTS SUMMARY =====")
        ana_logger.info(f"Samples attempted: {len(samples)}")
        ana_logger.info(f"Successful result entries: {len(results_df)}")
        if not processing_errors_df.empty:
            ana_logger.warning(f"Processing error entries: {len(processing_errors_df)}")

        # --- 6. Prepare Final DataFrame for CSV Saving ---
        full_df_to_save = pd.concat([results_df, processing_errors_df], ignore_index=True)

        # Serialize list/dict columns using the robust helper function
        potential_list_cols = ['sample_results', 'entities', 'unique_characters',
                            'emotional_sentences', 'dialogue_snippets']
        for col in potential_list_cols:
            if col in full_df_to_save.columns:
                dtype_kind = full_df_to_save[col].dtype.kind
                # Apply only if column is object type (potential lists/dicts/mixed)
                if dtype_kind == 'O':
                    ana_logger.debug(f"Applying JSON serialization logic to object column '{col}'.")
                    # Use the robust helper function
                    full_df_to_save[col] = full_df_to_save[col].apply(serialize_if_needed)

        # --- 7. Save Detailed Results CSV ---
        csv_file = results_dir / f"benchmark_results_{run_timestamp}.csv"
        full_df_to_save.to_csv(csv_file, index=False, encoding='utf-8')
        ana_logger.info(f"Detailed results (including errors) saved to: {csv_file}")

        # --- 8. Prepare and Save JSON Summary ---
        json_summary_file = results_dir / f"benchmark_summary_{run_timestamp}.json"
        config_details = {}
        try:
            # Access global constants safely
            config_details = {
                "corpus_dir": str(CORPUS_DIR), "spacy_model": SPACY_MODEL,
                "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "max_workers": MAX_WORKERS,
                "sentiment_polarity_threshold": SENTIMENT_POLARITY_THRESHOLD,
                "sentiment_subjectivity_threshold": SENTIMENT_SUBJECTIVITY_THRESHOLD
            }
        except NameError as ne:
            ana_logger.warning(f"Could not access global constants for config summary: {ne}. Saving limited config.")
            config_details = {"error": f"Global config constants not accessible ({ne})"}

        summary_data = {
            "run_timestamp": run_timestamp,
            "config": config_details,
            "samples_processed_count": len(samples),
            "total_result_entries": len(full_df_to_save),
            "successful_result_entries": len(results_df),
            "processing_error_entries": len(processing_errors_df),
            "performance_summary": summary.reset_index().to_dict(orient='records') if not summary.empty else []
        }
        with open(json_summary_file, "w", encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, default=str) # default=str handles numpy types
        ana_logger.info(f"Run summary saved to: {json_summary_file}")

        # --- 9. Return DataFrame of Successful Results ---
        return results_df

    except Exception as e:
        ana_logger.error(f"Analysis and saving failed unexpectedly: {format_error(e)}", exc_info=True)
        return None # Indicate failure


def create_visualizations(results_df: pd.DataFrame, results_dir: Path, run_timestamp: str) -> None:
    """
    Generates visualizations for benchmarking results and saves them as image files.

    Args:
        results_df (pd.DataFrame): DataFrame containing the benchmarking results.
        results_dir (Path): Directory where the visualizations will be saved.
        run_timestamp (str): Timestamp string used to uniquely identify the visualization files.

    Returns:
        None

    Notes:
        - The function validates the input data and logs warnings or errors for invalid or missing data.
        - It generates various plots, including bar plots, scatter plots, and box plots, to visualize the results.
        - The function ensures that the output directory exists and handles exceptions gracefully.
    """
    vis_logger = logging.getLogger("nlp_benchmark")
    if not isinstance(results_df, pd.DataFrame):
        vis_logger.error("Visualization skipped: Invalid DataFrame.")
        return
    if results_df.empty:
        vis_logger.warning("Visualization skipped: No data in DataFrame.")
        return
    if not isinstance(results_dir, Path):
        vis_logger.error("Visualization skipped: Invalid results directory.")
        return
    if not isinstance(run_timestamp, str):
        vis_logger.error("Visualization skipped: Invalid timestamp.")
        return

    vis_logger.info("Generating visualizations...")
    try:
        sns.set_theme(style="whitegrid")
        plot_dir = results_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Plot 1: Average Processing Time by Task and Library
        time_pivot = results_df.pivot_table(index='task', columns='library', values='time_taken', aggfunc='mean')
        if not time_pivot.empty:
            plt.figure(figsize=(12, 7))
            time_pivot.plot(kind='bar', ax=plt.gca())
            plt.title('Average Processing Time by Task and Library')
            plt.ylabel('Mean Time (s)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(plot_dir / f"time_by_task_{run_timestamp}.png", dpi=300)
            plt.close()
            vis_logger.info(f"Plot saved: time_by_task_{run_timestamp}.png")
        else:
            vis_logger.warning("Skipping plot time_by_task: No data available.")

        # Plot 2: Processing Speed Comparison
        speed_data = results_df[results_df['task'] == 'processing_speed'].copy()
        speed_data = speed_data.dropna(subset=['chars_per_second'])
        speed_data = speed_data[np.isfinite(speed_data['chars_per_second'])]
        if not speed_data.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='library', y='chars_per_second', data=speed_data, palette='viridis', ci=None)
            plt.title('Processing Speed Comparison (Chars/s)')
            plt.ylabel('Chars/s')
            plt.tight_layout()
            plt.savefig(plot_dir / f"speed_comparison_{run_timestamp}.png", dpi=300)
            plt.close()
            vis_logger.info(f"Plot saved: speed_comparison_{run_timestamp}.png")
        else:
            vis_logger.warning("Skipping plot speed_comparison: No data available.")

        # Plot 3: Processing Time vs Sample Length
        plot_data_time = results_df[results_df['merged'] == False] if 'merged' in results_df.columns and not results_df[results_df['merged'] == False].empty else results_df
        plot_data_time = plot_data_time.dropna(subset=['sample_length', 'time_taken'])
        if not plot_data_time.empty:
            plt.figure(figsize=(12, 7))
            sns.scatterplot(
                data=plot_data_time,
                x='sample_length',
                y='time_taken',
                hue='library',
                style='task',
                size='time_taken',
                sizes=(20, 200),
                alpha=0.7
            )
            plt.title('Processing Time vs Sample Length')
            plt.xlabel('Sample Length (chars)')
            plt.ylabel('Processing Time (s)')
            plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(plot_dir / f"time_vs_length_{run_timestamp}.png", dpi=300)
            plt.close()
            vis_logger.info(f"Plot saved: time_vs_length_{run_timestamp}.png")
        else:
            vis_logger.warning("Skipping plot time_vs_length: No data available.")

        # Plot 4: Distribution of Processing Time by Task and Library
        if not plot_data_time.empty:
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=plot_data_time, x='task', y='time_taken', hue='library', palette='pastel')
            plt.title('Distribution of Processing Time by Task and Library')
            plt.ylabel('Processing Time (s)')
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(plot_dir / f"time_distribution_boxplot_{run_timestamp}.png", dpi=300)
            plt.close()
            vis_logger.info(f"Plot saved: time_distribution_boxplot_{run_timestamp}.png")
        else:
            vis_logger.warning("Skipping plot time_distribution_boxplot: No data available.")

        vis_logger.info("Visualization generation complete.")
    except Exception as e:
        vis_logger.error(f"Visualization generation failed: {format_error(e)}", exc_info=True)

def validate_against_ground_truth(
    results_df: pd.DataFrame,
    samples: dict[str, str], # Keep samples if needed
    results_dir: Path,
    run_timestamp: str
) -> pd.DataFrame:
    """
    Compare benchmark results against ground truth data and calculate metrics.
    Includes check and warning for missing ground truth files.
    Handles missing data keys in results.
    Adjusts sample_id to look for ground truth files without '.txt'.

    Args:
        results_df: DataFrame with benchmark results.
        samples: Dictionary of sample IDs and original text.
        results_dir: Path to the results directory.
        run_timestamp: Timestamp string for file naming.

    Returns:
        DataFrame with validation metrics (precision, recall, F1) per sample/task/library.
        Returns an empty DataFrame if no metrics are generated.
    """
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        logger.warning("Validation skipped: Invalid or empty results DataFrame provided.")
        return pd.DataFrame()
    # Add other input checks if needed (results_dir, run_timestamp)

    validation_results = []
    # Ensure sample_id column exists before using unique()
    if 'sample_id' not in results_df.columns:
        logger.error("Validation skipped: 'sample_id' column missing in results DataFrame.")
        return pd.DataFrame()
    processed_sample_ids = results_df['sample_id'].unique()
    logger.info(f"Starting validation against ground truth for {len(processed_sample_ids)} unique sample IDs found in results...")

    gt_files_found_count = 0 # Counter for samples with GT

    for sample_id in processed_sample_ids:
        # --- Filename Fix: Remove .txt before loading GT ---
        if isinstance(sample_id, str) and sample_id.endswith(".txt"):
            gt_lookup_id = sample_id[:-4] # Remove last 4 chars (".txt")
        else:
            gt_lookup_id = sample_id # Use as is if no .txt extension
            logger.warning(f"Sample ID '{sample_id}' does not end with '.txt'. Using ID as is for ground truth lookup.")

        logger.debug(f"Attempting to load ground truth for lookup ID: '{gt_lookup_id}' (derived from sample ID: '{sample_id}')")
        gt = load_ground_truth(gt_lookup_id) # Use the modified ID

        # --- Check if GT was loaded ---
        if not gt: # Checks for None or empty dict potentially
            logger.warning(f"Ground truth file not found or failed to load for lookup ID '{gt_lookup_id}' (derived from sample '{sample_id}'). Skipping validation for this sample.")
            continue
        # --- End of Check ---

        gt_files_found_count += 1
        logger.debug(f"Processing validation for sample '{sample_id}' using loaded ground truth.")

        # Filter results for the current original sample_id
        sample_results_df = results_df[results_df['sample_id'] == sample_id].copy()

        for _, result_row in sample_results_df.iterrows():
            task = result_row.get('task')
            library = result_row.get('library')

            # Initialize metrics for appending later
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            metrics_calculated = False # Flag to track if calculation happened

            try:
                # --- NER Validation ---
                if task == 'ner':
                    gt_chars_list = gt.get('characters', [])
                    # Check for None only, as gt.get defaults to [] if key missing
                    if gt_chars_list is None: gt_chars_list = []
                    gt_chars: Set[str] = set(gt_chars_list)

                    detected_chars_list = result_row.get('unique_characters')
                    # Check for None only before isinstance check
                    if detected_chars_list is None: detected_chars_list = []

                    # Check if data is valid list before creating set
                    if isinstance(detected_chars_list, list):
                        detected_chars: Set[str] = set(detected_chars_list)
                        true_positives = len(gt_chars.intersection(detected_chars))
                        false_positives = len(detected_chars - gt_chars)
                        false_negatives = len(gt_chars - detected_chars)
                        metrics_calculated = True
                    else:
                        logger.warning(f"Missing or invalid 'unique_characters' list data for NER validation: {sample_id}/{library}")
                        false_negatives = len(gt_chars) # All GT chars are missed

                # --- Dialogue Detection Validation ---
                elif task == 'dialogue_detection':
                    gt_dialogues_list = gt.get('dialogues', [])
                    if gt_dialogues_list is None: gt_dialogues_list = []
                    gt_dialogues: Set[str] = set(gt_dialogues_list)

                    results_data = result_row.get('sample_results')
                    # Check for None only before isinstance check
                    if results_data is None: results_data = []

                    detected_texts: Set[str] = set()
                    if isinstance(results_data, list):
                        # Filter items robustly before accessing index [1]
                        valid_items = [item for item in results_data if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], str)]
                        detected_texts = set(item[1] for item in valid_items)
                        true_positives = len(gt_dialogues.intersection(detected_texts))
                        false_positives = len(detected_texts - gt_dialogues)
                        false_negatives = len(gt_dialogues - detected_texts)
                        metrics_calculated = True
                    else:
                        logger.warning(f"Missing or invalid 'sample_results' list data for Dialogue validation: {sample_id}/{library}")
                        false_negatives = len(gt_dialogues) # All GT dialogues are missed

                # --- Sentiment Validation ---
                elif task == 'sentiment':
                    gt_emotions_list = gt.get('emotions', [])
                    if gt_emotions_list is None: gt_emotions_list = []
                    # Extract sentences from GT [sentence, score] pairs robustly
                    gt_sentences: Set[str] = set(item[0] for item in gt_emotions_list if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str))

                    results_data = result_row.get('sample_results')
                    # Check for None only before isinstance check
                    if results_data is None: results_data = []

                    detected_sentences: Set[str] = set()
                    if isinstance(results_data, list):
                        # Extract sentences from detected [sentence, score] pairs robustly
                        valid_items = [item for item in results_data if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str)]
                        detected_sentences = set(item[0] for item in valid_items)
                        true_positives = len(gt_sentences.intersection(detected_sentences))
                        false_positives = len(detected_sentences - gt_sentences)
                        false_negatives = len(gt_sentences - detected_sentences)
                        metrics_calculated = True
                    else:
                        logger.warning(f"Missing or invalid 'sample_results' list data for Sentiment validation: {sample_id}/{library}")
                        false_negatives = len(gt_sentences) # All GT sentences are missed

                # --- Calculate Metrics (Common for all validated tasks) ---
                else:
                    continue # Skip tasks not being validated (e.g., processing_speed)

                # Calculate P/R/F1 only if metrics were calculated
                if metrics_calculated:
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                # If not calculated (due to missing data warning), P/R/F1 remain 0.0

                # Append results (even if P/R/F1 are 0 due to missing data/no overlap)
                validation_results.append({
                    "sample_id": sample_id,
                    "task": task,
                    "library": library,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                })

            except Exception as val_err:
                logger.error(f"Error during metric calculation for {sample_id}/{task}/{library}: {val_err}", exc_info=True)
                # Skip appending metrics for this specific error case
                continue


    # --- Finalize and Save ---
    if not validation_results:
        if gt_files_found_count > 0:
            logger.warning(f"Validation attempted for {gt_files_found_count} samples with ground truth, but no valid metrics could be generated (check warnings above, validation logic, result keys, or GT content).")
        else:
            logger.warning("No ground truth files were found or loaded for any processed samples. No validation metrics generated.")
        return pd.DataFrame()

    validation_df = pd.DataFrame(validation_results)
    # Check if df is empty again after potential errors during calculation
    if validation_df.empty:
        logger.warning("Validation DataFrame is empty after processing attempts. No metrics to save.")
        return pd.DataFrame()

    logger.info(f"Validation complete. Generated {len(validation_df)} metric records for {gt_files_found_count} samples.")

    # Save validation results CSV
    try:
        csv_filepath = results_dir / f"validation_results_{run_timestamp}.csv"
        validation_df.to_csv(csv_filepath, index=False, encoding='utf-8')
        logger.info(f"Validation results saved to: {csv_filepath}")
    except Exception as e:
        logger.error(f"Error saving validation results CSV: {format_error(e)}", exc_info=True) # Assuming format_error exists

    # Create and save validation visualization (F1 Score)
    try:
        if not validation_df.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='library', y='f1', hue='task', data=validation_df, palette='viridis', errorbar=None)
            plt.title(f'F1 Score by Library and Task ({run_timestamp})')
            plt.ylim(0, 1.05) # Set y-axis limits
            plt.ylabel('F1 Score')
            plt.xlabel('Library')
            plt.legend(title='Task')
            plt.tight_layout()
            plot_filepath = results_dir / "plots" / f"validation_f1_{run_timestamp}.png"
            plot_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure plot directory exists
            plt.savefig(plot_filepath, dpi=300)
            logger.info(f"Validation F1 score visualization saved to: {plot_filepath}")
            plt.close() # Close the plot to free memory
        else:
            logger.warning("Skipping validation plot generation as no metrics were calculated or saved.")
    except Exception as e:
        logger.error(f"Error visualizing validation results: {format_error(e)}", exc_info=True) # Assuming format_error exists

    return validation_df


# --- Cleanup Functions ---
def remove_temp_files(directory: Path = CORPUS_DIR, gt_directory: Path = GROUND_TRUTH_DIR) -> None:
    """
    Removes temporary dummy corpus and ground truth directories if they appear to 
    contain only auto-generated dummy data. This function performs basic checks 
    to identify such directories and logs the actions taken. Use with caution, 
    as incorrect identification may lead to unintended data loss.
    Args:
        directory (Path): The path to the corpus directory to check and potentially remove.
                        Defaults to CORPUS_DIR.
        gt_directory (Path): The path to the ground truth directory to check and potentially remove.
                            Defaults to GROUND_TRUTH_DIR.
    Behavior:
        - Checks if the `directory` exists and is named "corpus".
        - Verifies if the files in the `directory` match the expected dummy file names 
        (e.g., "sample1.txt", "sample2.txt").
        - If identified as dummy data, logs a warning and optionally removes the directory.
        - Similarly checks the `gt_directory` for expected dummy ground truth files 
        (e.g., "sample1_txt.json", "sample2_txt.json").
        - Logs actions and skips removal if the directories do not meet the criteria.
    Logging:
        - Logs informational messages about the cleanup process.
        - Logs warnings when directories are identified as potentially deletable.
        - Logs errors if exceptions occur during the cleanup process.
    Note:
        - The actual removal of directories is commented out for safety. Uncomment the 
        `shutil.rmtree` lines to enable deletion after thorough testing.
        - Adjust the expected file names or content checks as needed to match the 
        specific dummy data patterns in your use case.
    """
    logger = logging.getLogger("nlp_benchmark")  # Ensure logger is defined
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(funcName)s - %(message)s"  # noqa: E501
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.info("Attempting cleanup of temporary dummy data...")
    # Basic check: Does the directory exist and is it named 'corpus'?
    # A more robust check would involve specific file names or content hashes.
    is_dummy_corpus = directory.name == "corpus" and directory.exists()
    # Example check: are there only files named sample1.txt, sample2.txt?
    try:
        if is_dummy_corpus:
            corpus_files = [f.name for f in directory.iterdir() if f.is_file()]
            if set(corpus_files) == {"sample1.txt", "sample2.txt"}: # Check content?
                logger.warning(f"Identified '{directory}' as potentially deletable dummy data.")
                # shutil.rmtree(directory) # Uncomment carefully!
                # logger.info(f"Removed dummy corpus directory: {directory}")

                # Also attempt to remove dummy ground truth if it matches
                if gt_directory.exists():
                    gt_files = [f.name for f in gt_directory.iterdir() if f.is_file()]
                    expected_gt = {"sample1_txt.json", "sample2_txt.json"} # Adjust if naming changes
                    if set(gt_files).issubset(expected_gt): # Check if GT files seem to be the dummy ones
                        logger.warning(f"Identified '{gt_directory}' as potentially deletable dummy ground truth.")
                        # shutil.rmtree(gt_directory) # Uncomment carefully!
                        # logger.info(f"Removed dummy ground truth directory: {gt_directory}")
                    else:
                        logger.info(f"Ground truth directory '{gt_directory}' does not contain expected dummy files. Skipping removal.")
            else:
                logger.info(f"Directory '{directory}' exists but does not contain expected dummy files. Skipping removal.")
        else:
            logger.info(f"Directory '{directory}' not identified as deletable dummy data. Skipping removal.")

    except Exception as e:
        logger.error(f"Error during temporary file cleanup: {format_error(e)}")


def close_database_connections() -> None:
    """Placeholder for closing database connections if used."""
    # Add actual DB closing logic here if applicable
    logger.debug("Closing database connections (placeholder).")

def release_resources() -> None:
    """
    Releases resources such as GPU memory, database connections, and other system resources.

    This function ensures proper cleanup of resources to prevent memory leaks or resource locking.
    It handles CUDA cache clearing, multiprocessing cleanup, and database connections closure.
    Each cleanup operation is performed in a separate try-except block for robustness.
    """
    release_logger = logging.getLogger("nlp_benchmark")
    release_logger.info("Releasing resources...")

    # Clear CUDA cache if PyTorch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            release_logger.info("Cleared PyTorch CUDA cache.")
    except ImportError:
        release_logger.debug("PyTorch not found, skipping CUDA cache clearing.")
    except Exception as e:
        release_logger.warning(f"Error clearing CUDA cache: {format_error(e)}")

    # Release multiprocessing resources
    try:
        active_processes = mp.active_children()
        for process in active_processes:
            process.terminate()
            process.join(timeout=1.0)  # Wait up to 1 second for each process
        release_logger.info(f"Terminated {len(active_processes)} active multiprocessing child processes.")
    except Exception as e:
        release_logger.warning(f"Error terminating multiprocessing resources: {format_error(e)}")

    # Close database connections if any exist
    try:
        close_database_connections()
        release_logger.info("Database connections closed successfully.")
    except Exception as e:
        release_logger.warning(f"Error closing database connections: {format_error(e)}")

    # Clear memory explicitly
    try:
        clear_memory()
        release_logger.info("Garbage collection triggered successfully.")
    except Exception as e:
        release_logger.warning(f"Error during garbage collection: {format_error(e)}")

    release_logger.info("Resource release completed.")

# --- Dummy Sample Data Function ---
def get_dummy_samples_data() -> dict:
    """
    Returns a dictionary containing dummy sample data for testing purposes.
    """
    return {
        "sample1.txt": 'This is a great movie! "I absolutely loved it," said Alice. "But the ending was so sad," Bob whispered.',
        "sample2.txt": "Dr. John Smith went to London. He met Ms. Jane Doe. \"The weather is dreary,\" John complained. They visited Big Ben.",
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure logger is initialized before use (assuming setup from original script)
    if 'logger' not in globals():
        # Basic logger setup if not already done (copy from original)
        logger = logging.getLogger("nlp_benchmark")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(processName)s - %(levelname)s - %(funcName)s - %(message)s"
            )
            # File handler (ensure RESULTS_DIR is defined)
            # fh = logging.FileHandler(LOG_FILE, mode='w')
            # fh.setFormatter(formatter)
            # logger.addHandler(fh)
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    start_run_time = time.monotonic() # Start timer
    logger.info(f"===== Starting NLP Benchmark Run ({datetime.datetime.now()}) =====")
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Setup: NLTK Resources & Dummy Data (if needed) ---
    try:
        download_nltk_resources() # Ensure NLTK resources are ready
    except Exception as nltk_e:
        logger.critical(f"Failed to prepare NLTK resources: {nltk_e}. Exiting.")
        sys.exit(1)

    dummy_data_created = False
    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.rglob("*.txt")):
        logger.warning(f"Corpus directory '{CORPUS_DIR}' empty or not found. Creating dummy data and ground truth.")
        try:
            CORPUS_DIR.mkdir(parents=True, exist_ok=True)
            GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

            dummy_samples_data = get_dummy_samples_data() # Fetch dummy data

            # Write dummy sample files
            for filename, text in dummy_samples_data.items():
                with open(CORPUS_DIR / filename, "w", encoding="utf-8") as f:
                    f.write(text)

            # Create corresponding ground truth sequentially
            gt1_success = create_ground_truth(
                sample_id="sample1.txt", # Use relative path as ID
                text=dummy_samples_data["sample1.txt"],
                dialogues=["I absolutely loved it,", "But the ending was so sad,"], # Example GT
                characters=["Alice", "Bob"],
                emotions=[("This is a great movie!", 0.62)] # Example sentiment GT
            )

            gt2_success = create_ground_truth(
                sample_id="sample2.txt",
                text=dummy_samples_data["sample2.txt"],
                dialogues=["The weather is dreary,"],
                characters=["John Smith", "Jane Doe", "John"], # Example GT
                emotions=[("The weather is dreary,", -0.34)]
            )

            if gt1_success and gt2_success:
                logger.info("Dummy data and ground truth created successfully.")
                dummy_data_created = True # Flag that dummy data was used
            else:
                logger.error("Failed to create dummy ground truth files.")
                # Decide if execution should stop if GT creation fails
                # sys.exit(1)

        except Exception as setup_e:
            logger.critical(f"Failed during dummy data setup: {format_error(setup_e)}", exc_info=True)
            sys.exit(1)

    # --- Load Samples ---
    samples_data = load_samples(directory=CORPUS_DIR)
    if not samples_data:
        logger.critical("No samples loaded. Benchmark run cannot proceed. Exiting.")
        sys.exit(1) # Exit if loading failed

    # --- Process Samples ---
    # Ensure RESULTS_DIR is defined and is a Path object
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure results dir exists

    all_run_results = process_multiple_samples(samples_data)

    # --- Analyze and Save Results ---
    results_dataframe = None # Initialize
    if all_run_results:
        results_dataframe = analyze_and_save_results(
            all_run_results, samples_data, RESULTS_DIR, current_timestamp
        )
    else:
        logger.warning("No results were generated from processing samples. Skipping analysis and visualization.")

    # --- Validate Against Ground Truth ---
    validation_dataframe = None # Initialize
    if results_dataframe is not None and not results_dataframe.empty:
        # Filter for merged results before validation if the flag exists and is meaningful
        # If chunking is not used, all results might have merged=False or chunk_position='full_sample'
        # Adjust filter as needed based on expected output structure
        if 'merged' in results_dataframe.columns:
            results_to_validate = results_dataframe[results_dataframe['merged'] == True].copy()
            if results_to_validate.empty:
                logger.warning("No 'merged' results found in the dataframe. Attempting validation on all results.")
                results_to_validate = results_dataframe # Fallback to all results if no merged ones
        else:
            logger.warning("'merged' column not found. Attempting validation on all results.")
            results_to_validate = results_dataframe # Validate all if no merge flag

        if not results_to_validate.empty:
            logger.info("Attempting validation against ground truth...")
            validation_dataframe = validate_against_ground_truth(
                results_to_validate, samples_data, RESULTS_DIR, current_timestamp
            )
            if validation_dataframe is not None and not validation_dataframe.empty:
                logger.info("Validation completed.")
            else:
                logger.warning("Validation did not produce any results.")
        else:
            logger.warning("No suitable results found in the dataframe to validate.")
    else:
        logger.warning("Skipping validation as benchmark analysis failed or produced no results.")

    # --- Create Visualizations (Based on Full Analysis Results) ---
    if results_dataframe is not None and not results_dataframe.empty:
        logger.info("Attempting to create visualizations...")
        create_visualizations(results_dataframe, RESULTS_DIR, current_timestamp)
        logger.info("Visualization creation process finished.")
    else:
        logger.warning("Skipping visualization creation as analysis failed or produced no results.")


    # --- Final Cleanup ---
    logger.info("Performing final cleanup...")
    # remove_temp_files(CORPUS_DIR, GROUND_TRUTH_DIR) # Be cautious enabling this! Only enable if dummy_data_created is True?
    if dummy_data_created:
        logger.warning("Dummy data was created for this run. Consider manual review before removing.")
        # Add logic here if you want to automatically remove dummy data based on the flag.
        # remove_temp_files(CORPUS_DIR, GROUND_TRUTH_DIR) # Example: Uncomment to remove if dummy data was made.

    close_database_connections() # Placeholder
    release_resources() # Attempt to release GPU mem, etc.

    end_run_time = time.monotonic()
    total_time = end_run_time - start_run_time
    logger.info(f"===== Benchmark Run Finished ({datetime.datetime.now()}) =====")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    try:
        # Ensure RESULTS_DIR is accessible here
        logger.info(f"Results, logs, and plots saved in: {RESULTS_DIR.resolve()}")
    except Exception as e:
        logger.error(f"Error resolving final RESULTS_DIR path: {format_error(e)}")

    logger.info("===== END OF SCRIPT =====")