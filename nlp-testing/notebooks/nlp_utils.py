import datetime
import os
import time
import json
import logging
import gc
import nltk
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil  # Import the shutil module
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer  # Keep this import

# --- NLTK Downloads ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
     nltk.data.find('taggers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
try:
     nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

# Constants - Keep these here!
CORPUS_DIR = "../data/corpus"
GROUND_TRUTH_DIR = "../data/ground_truth"
RESULTS_DIR = "../results"
LOG_FILE = "benchmark.log"
CHUNK_SIZE = 10000
OVERLAP = 1000
MAX_WORKERS = 6


# --- Logging Setup --- (Keep the logging setup here!)
logger = logging.getLogger("nlp_benchmark")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
)

# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler (for notebook output) -  We'll handle this differently in the notebook
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# logger.addHandler(ch)

nlp_lg = None  # Keep this here


def clear_memory():
    """Explicitly clear memory."""
    gc.collect()


def init_worker():
    """Initialize worker process."""
    global nlp_lg
    try:
        spacy.prefer_gpu()  # Try to use GPU.  MUST be before spacy.load()
        nlp_lg = spacy.load("en_core_web_trf")  # Load the transformer model!
        logger.info(f"Worker {os.getpid()}: spaCy model loaded.")
        if spacy.util.has_gpu():
            logger.info(f"Worker {os.getpid()}: spaCy model loaded on GPU.")
        else:
            logger.warning(f"Worker {os.getpid()}: spaCy model loaded on CPU.")

        if "spacytextblob" not in nlp_lg.pipe_names:
            from spacytextblob.spacytextblob import SpacyTextBlob
            nlp_lg.add_pipe("spacytextblob")
            logger.info(f"Worker {os.getpid()}: spacytextblob added to pipeline.")

    except OSError as e:
        logger.error(f"Worker {os.getpid()}: Could not load spacy model: {e}.")
        raise
    except Exception as e:
        logger.error(f"Worker {os.getpid()}: Error during initialization: {e}")
        raise


def ner_spacy(text: str, nlp: spacy.language.Language) -> dict:
    """
    Extract named entities (specifically people) using spaCy.

    Args:
        text: The input text.
        nlp: The spaCy language model instance.  Pass this in!

    Returns:
        A dictionary containing the extracted entities, unique characters,
        processing time, and the count of unique characters.

    Raises:
        TypeError: If 'text' is not a string or 'nlp' is not a spaCy Language object.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(nlp, spacy.language.Language):
        raise TypeError("Input 'nlp' must be a spaCy Language object.")

    start_time = time.time()
    try:
        doc = nlp(text)

        # Extract PERSON entities as potential characters
        entities = [
            (ent.text, ent.label_) for ent in doc.ents if ent.label_ == "PERSON"
        ]

        # Count unique character names
        unique_characters = set(entity[0] for entity in entities)

        end_time = time.time()
        return {
            "entities": entities,
            "unique_characters": list(unique_characters),
            "time_taken": end_time - start_time,
            "count": len(unique_characters),
        }
    except Exception as e:
        logger.error(f"Error in ner_spacy: {e}")
        return {
            "entities": [],
            "unique_characters": [],
            "time_taken": time.time() - start_time,  # Still record time
            "count": 0,
            "error": str(e),  # Include the error message
        }


def ner_nltk(text: str) -> dict:
    """
    Extract named entities (specifically people) using NLTK.

    Args:
        text: The input text.

    Returns:
        A dictionary containing the extracted entities, unique characters,
        processing time, and the count of unique characters.

    Raises:
        TypeError: If 'text' is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")

    start_time = time.time()
    try:
        entities = []
        sentences = nltk.sent_tokenize(text)

        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(tagged)

            # Extract person entities
            for chunk in chunks:
                if hasattr(chunk, "label") and chunk.label() == "PERSON":
                    name = " ".join(c[0] for c in chunk)
                    entities.append((name, "PERSON"))

        # Count unique character names
        unique_characters = set(entity[0] for entity in entities)

        end_time = time.time()
        return {
            "entities": entities,
            "unique_characters": list(unique_characters),
            "time_taken": end_time - start_time,
            "count": len(unique_characters),
        }
    except LookupError as e:
        logger.error(
            f"NLTK resource missing: {e}. Please download the required resources."
        )
        return {
            "entities": [],
            "unique_characters": [],
            "time_taken": time.time() - start_time,
            "count": 0,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in ner_nltk: {e}")
        return {  # consistent return
            "entities": [],
            "unique_characters": [],
            "time_taken": time.time() - start_time,
            "count": 0,
            "error": str(e),
        }

# Cell 5: Sentiment Analysis
def sentiment_spacy(text: str, nlp: spacy.language.Language) -> dict:
    """
    Analyze sentiment using spaCy with SpacyTextBlob.

    Args:
        text: The input text.
        nlp: The spaCy language model instance.

    Returns:
        A dictionary containing the top emotional sentences, processing time,
        and the count of emotional sentences.

    Raises:
        TypeError: If 'text' is not a string or 'nlp' is not a spaCy Language object.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(nlp, spacy.language.Language):
        raise TypeError("Input 'nlp' must be a spaCy Language object.")

    start_time = time.time()
    try:
        doc = nlp(text)

        # Get sentence-level sentiment
        sentences = [sent.text for sent in doc.sents]
        emotional_sentences = []

        for sent in sentences:
            sent_doc = nlp(sent)
            # SpacyTextBlob provides polarity and subjectivity
            polarity = sent_doc._.blob.polarity
            subjectivity = sent_doc._.blob.subjectivity

            # Consider sentences with non-zero polarity and high subjectivity
            if abs(polarity) > 0.1 and subjectivity > 0.3:
                emotional_sentences.append((sent, polarity))

        # Sort by emotional intensity (absolute value of polarity)
        emotional_sentences.sort(key=lambda x: abs(x[1]), reverse=True)

        end_time = time.time()
        return {
            "emotional_sentences": emotional_sentences[:5],  # Top 5
            "time_taken": end_time - start_time,
            "count": len(emotional_sentences),
        }
    except Exception as e:
        logger.error(f"Error in sentiment_spacy: {e}")
        return {
            "emotional_sentences": [],
            "time_taken": time.time() - start_time,
            "count": 0,
            "error": str(e),
        }


def sentiment_nltk(text: str) -> dict:
    """
    Analyze sentiment using NLTK's VADER.

    Args:
        text: The input text.

    Returns:
        A dictionary containing the top emotional sentences, processing time,
        and the count of sentences analyzed.

    Raises:
        TypeError: If 'text' is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")

    start_time = time.time()
    try:
        sid = SentimentIntensityAnalyzer()
        sentences = nltk.sent_tokenize(text)
        sentiment_scores = [sid.polarity_scores(sentence) for sentence in sentences]

        # Find most emotional sentences (highest absolute compound score)
        sentence_sentiments = [
            (sentences[i], scores["compound"])
            for i, scores in enumerate(sentiment_scores)
        ]
        emotional_sentences = sorted(
            sentence_sentiments, key=lambda x: abs(x[1]), reverse=True
        )[:5]

        end_time = time.time()
        return {
            "emotional_sentences": emotional_sentences,
            "time_taken": end_time - start_time,
            "count": len(sentence_sentiments),
        }

    except LookupError as e:
        logger.error(
            f"NLTK resource missing: {e}. Please download the required resources."
        )
        return {
            "emotional_sentences": [],
            "time_taken": time.time() - start_time,
            "count": 0,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in sentiment_nltk: {e}")
        return {  # Consistent return structure
            "emotional_sentences": [],
            "time_taken": time.time() - start_time,
            "count": 0,
            "error": str(e),
        }


# Cell 6: Processing Speed Functions
def speed_spacy(text: str, nlp: spacy.language.Language) -> dict:
    """
    Measure processing speed using spaCy.

    Args:
        text: The input text.
        nlp: The spaCy language model instance.

    Returns:
        A dictionary containing processing time, characters per second, and
        the text length.

    Raises:
        TypeError: If 'text' is not a string or 'nlp' is not a spaCy Language object.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(nlp, spacy.language.Language):
        raise TypeError("Input 'nlp' must be a spaCy Language object.")

    start_time = time.time()
    try:
        doc = nlp(text)

        # Perform some standard NLP operations
        _ = [token.text for token in doc]  # Tokenization
        _ = [token.pos_ for token in doc]  # POS tagging
        _ = [token.dep_ for token in doc]  # Dependency parsing
        _ = [ent for ent in doc.ents]  # NER

        end_time = time.time()
        processing_time = end_time - start_time
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0

        return {
            "time_taken": processing_time,
            "chars_per_second": chars_per_second,
            "text_length": len(text),
        }
    except Exception as e:
        logger.error(f"Error in speed_spacy: {e}")
        return {
            "time_taken": 0,  # Return 0 for time_taken on error
            "chars_per_second": 0,
            "text_length": len(text),
            "error": str(e),
        }


def speed_nltk(text: str) -> dict:
    """
    Measure processing speed using NLTK.

    Args:
        text: The input text.

    Returns:
        A dictionary containing processing time, characters per second, and
        the text length.

    Raises:
        TypeError: If 'text' is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")

    start_time = time.time()
    try:
        sentences = nltk.sent_tokenize(text)
        all_tokens = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            all_tokens.extend(tokens)

        # Limit to first 10,000 tokens for POS tagging
        tagged = nltk.pos_tag(all_tokens[:10000])

        end_time = time.time()
        processing_time = end_time - start_time
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0

        return {
            "time_taken": processing_time,
            "chars_per_second": chars_per_second,
            "text_length": len(text),
        }
    except LookupError as e:
        logger.error(
            f"NLTK resource missing: {e}. Please download the required resources."
        )
        return {
            "time_taken": 0,
            "chars_per_second": 0,
            "text_length": len(text),
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in speed_nltk: {e}")
        return {  # Consistent return structure
            "time_taken": 0,
            "chars_per_second": 0,
            "text_length": len(text),
            "error": str(e),
        }


# Cell 7: Dialogue Detection Functions (Revised)
def dialogue_detection_spacy(text: str, nlp: spacy.language.Language) -> dict:
    """
    Detect dialogue in text using spaCy.

    Args:
        text: The input text.
        nlp: The spaCy language model instance.

    Returns:
        A dictionary containing the detected dialogue sentences, processing time,
        characters per second, and text length.

    Raises:
        TypeError: If input types are incorrect.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(nlp, spacy.language.Language):
        raise TypeError("Input 'nlp' must be a spaCy Language object.")

    start_time = time.time()
    try:
        doc = nlp(text)

        # More robust dialogue detection:
        dialogue_sentences = []
        for sent in doc.sents:
            # Check for balanced quotes and sufficient length.
            if sent.text.count('"') >= 2 and len(sent.text) > 2:
                # Extract text between first and last quote.
                start_quote = sent.text.find('"')
                end_quote = sent.text.rfind('"')
                if start_quote != end_quote:  # ensure not the same
                    dialogue_sentences.append(sent.text[start_quote + 1 : end_quote])

        end_time = time.time()
        processing_time = end_time - start_time
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0

        return {
            "dialogue_sentences": dialogue_sentences,
            "time_taken": processing_time,
            "chars_per_second": chars_per_second,
            "text_length": len(text),
            "count": len(dialogue_sentences),  # added for consistency
        }
    except Exception as e:
        logger.error(f"Error in dialogue_detection_spacy: {e}")
        return {
            "dialogue_sentences": [],
            "time_taken": 0,
            "chars_per_second": 0,
            "text_length": len(text),
            "count": 0,
            "error": str(e),
        }


def dialogue_detection_nltk(text: str) -> dict:
    """
    Detect dialogue in text using NLTK.

    Args:
        text: The input text.

    Returns:
        A dictionary containing the detected dialogue sentences, processing time,
        characters per second, and text length.

    Raises:
        TypeError: If input type is incorrect.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")

    start_time = time.time()
    try:
        sentences = nltk.sent_tokenize(text)
        dialogue_sentences = []
        for sent in sentences:
            # Check for balanced quotes and sufficient length
            if sent.count('"') >= 2 and len(sent) > 2:
                # Extract text between first and last quote
                start_quote = sent.find('"')
                end_quote = sent.rfind('"')
                if start_quote != end_quote:
                    dialogue_sentences.append(sent[start_quote + 1 : end_quote])

        end_time = time.time()
        processing_time = end_time - start_time
        chars_per_second = len(text) / processing_time if processing_time > 0 else 0

        return {
            "dialogue_sentences": dialogue_sentences,
            "time_taken": processing_time,
            "chars_per_second": chars_per_second,
            "text_length": len(text),
            "count": len(dialogue_sentences),
        }
    except Exception as e:
        logger.error(f"Error in dialogue_detection_nltk: {e}")
        return {  # Consistent return structure
            "dialogue_sentences": [],
            "time_taken": 0,
            "chars_per_second": 0,
            "text_length": len(text),
            "count": 0,
            "error": str(e),
        }


# Cell 8: Benchmark Performance Function
def benchmark_performance(text: str, task: str = "all", libraries: list[str] | None = None) -> dict:
    """
    Benchmark NLP tasks performance on the given text.

    Args:
        text: The text to process.
        task: Task ("ner", "sentiment", etc., "all", or "minimal").
        libraries: Libraries to use ("spacy", "nltk").

    Returns:
        Results by task and library.
    """
    if libraries is None:
        libraries = ["spacy", "nltk"]

    results: dict[str, dict[str, dict]] = {}
    for t in (["ner", "sentiment", "dialogue_detection", "processing_speed"] if task == "all" else [task] if task != "minimal" else ["processing_speed"]):
        results[t] = {}
        for lib in libraries:
            try:
                if t == "ner":
                    results[t][lib] = ner_spacy(text, nlp_lg) if lib == "spacy" else ner_nltk(text)
                elif t == "sentiment":
                    results[t][lib] = sentiment_spacy(text, nlp_lg) if lib == "spacy" else sentiment_nltk(text)
                elif t == "dialogue_detection":
                    results[t][lib] = dialogue_detection_spacy(text, nlp_lg) if lib == "spacy" else dialogue_detection_nltk(text)
                elif t == "processing_speed":
                    results[t][lib] = speed_spacy(text, nlp_lg) if lib == "spacy" else speed_nltk(text)
            except Exception as e:
                results[t][lib] = {"time_taken": 0, "error": str(e)}
    return results

# Cell 9: Core Processing Function (Refactored into _process_chunk_or_sample

def _process_chunk_or_sample(sample_id: str, text: str, chunk_id: str | None = None, chunk_num: int | None = None, total_chunks: int | None = None) -> list[dict]:
    """
    Internal function to process either a single chunk or a complete sample.

    Args:
        sample_id: Identifier for the sample.
        text: The text content to process (either a chunk or the full sample).
        chunk_id: Identifier for the chunk (if processing a chunk).
        chunk_num: The chunk number (if processing a chunk).
        total_chunks: Total number of chunks (if processing a chunk).

    Returns:
        A list of result dictionaries.

    Raises:
        TypeError: if input types are incorrect.
    """
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_id is not None and not isinstance(chunk_id, str):
        raise TypeError("chunk_id must be a string or None")
    if chunk_num is not None and not isinstance(chunk_num, int):
        raise TypeError("chunk_num must be an integer or None")
    if total_chunks is not None and not isinstance(total_chunks, int):
        raise TypeError("total_chunks must be an integer or None")

    try:
        results = benchmark_performance(
            text=text,
            task="all",
            libraries=["spacy", "nltk"]
        )
        # ADD THESE LINES for GPU check:
        if nlp_lg.device_type == 'cuda':  # nlp_lg is accessible within this function
            logger.info(f"Processing {sample_id=}, {chunk_id=} on GPU.")
        else:
            logger.warning(f"Processing {sample_id=}, {chunk_id=} on CPU.")

    except MemoryError:
        logger.warning(f"Memory error processing {sample_id=}, {chunk_id=}. Retrying with minimal processing.")
        gc.collect()
        results = benchmark_performance(
            text=text,
            task="minimal",
            libraries=["nltk"]
        )
    except Exception as e:
        logger.error(f"Error processing {sample_id=}, {chunk_id=}: {e}")
        return [{
            "sample_id": sample_id,
            "chunk_id": chunk_id,
            "chunk_num": chunk_num,
            "chunk_size": len(text) if chunk_id else None,
            "sample_length": len(text),
            "error": str(e)
        }]

    processed_results = []
    for task, lib_results in results.items():
        for lib, res in lib_results.items():
            result_row = {
                "sample_id": sample_id,
                "sample_length": len(text),
                "task": task,
                "library": lib,
                "time_taken": res["time_taken"]
            }
            if chunk_id:
                result_row.update({
                    "chunk_id": chunk_id,
                    "chunk_num": chunk_num,
                    "chunk_size": len(text),
                    "chunk_position": "middle"
                })
                if chunk_num == 1:
                    result_row["chunk_position"] = "first"
                elif chunk_num == total_chunks:
                    result_row["chunk_position"] = "last"

            if task == "dialogue_detection" and "count" in res:
                result_row["count"] = res["count"]
                result_row["sample_dialogues"] = res["dialogue_sentences"][:3] if res["dialogue_sentences"] else []
            elif task == "ner" and "count" in res:
                result_row["unique_count"] = res["count"]
                result_row["characters"] = res["unique_characters"][:10] if res["unique_characters"] else []
            elif task == "sentiment" and "count" in res:
                result_row["emotional_count"] = res["count"]
                result_row["emotional_sentences"] = res["emotional_sentences"][:3] if res["emotional_sentences"] else []
            elif task == "processing_speed" and "chars_per_second" in res:
                result_row["chars_per_second"] = res["chars_per_second"]

            processed_results.append(result_row)

    return processed_results

# Cell 10: Sample and Chunk Processing Functions (Revised)
def process_sample_in_chunks(sample_id: str, sample_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """
    Process a sample text in smaller, overlapping chunks, or directly if small.

    Args:
        sample_id: Identifier for the sample.
        sample_text: The full text to process.
        chunk_size: Size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of processed results from all chunks (or the single sample).
        Empty list if sample_text is empty.
    """
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    if not isinstance(sample_text, str):
        raise TypeError("sample_text must be a string")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if not sample_text:  # Handle empty sample text
        logger.warning(f"Sample {sample_id} is empty.")
        return []


    logger.info(f"Processing {sample_id} (length: {len(sample_text)} chars)")

    # Process the entire text if it's small enough
    if len(sample_text) <= chunk_size:
        return _process_chunk_or_sample(sample_id, sample_text)

    # Split into chunks for larger texts
    chunks = []
    chunk_ids = []
    for i in range(0, len(sample_text), chunk_size - overlap):
        chunk_end = min(i + chunk_size, len(sample_text))
        chunks.append(sample_text[i:chunk_end])
        chunk_ids.append(f"{sample_id}#chunk{len(chunks)}")

    logger.info(f"Split {sample_id} into {len(chunks)} chunks")

    sample_results = []
    for i, (chunk, chunk_id) in enumerate(tqdm(zip(chunks, chunk_ids),
                                              total=len(chunks),
                                              desc=f"Processing {sample_id}",
                                              unit="chunk"), 1):  # Start counter at 1
        try:
            chunk_results = _process_chunk_or_sample(
                sample_id, chunk, chunk_id, i, len(chunks)
            )
            sample_results.extend(chunk_results)
        except Exception as e:  # General exception, already handled in _process...
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            # No need to append here, as the error is handled in the sub-function.
            continue # Explicitly continue

        gc.collect() # explicit garbage collection

    return merge_chunk_results(sample_results)  # Moved merge_chunk_results here

def load_samples(directory: str = CORPUS_DIR) -> dict[str, str]:
    """Loads text samples from the specified directory."""
    samples = {}
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        sample_id = os.path.relpath(filepath, directory).replace("\\", "/")
                        samples[sample_id] = f.read()
    except FileNotFoundError:
        logger.error(f"Corpus directory not found: {directory}")
        return {}  # Return empty dict if directory doesn't exist
    except Exception as e:
        logger.error(f"An error occurred while loading samples: {e}")
        return {}
    return samples

# Cell 11: Result Merging Function (from original Cell 10)
def merge_chunk_results(results):
    """
    Merge and deduplicate results from overlapping chunks

    Args:
        results (list): List of result dictionaries from chunks

    Returns:
        list: Merged results with duplicates removed
    """
    if not results:
        return []

    # Group results by task and library
    grouped = defaultdict(list)

    for res in results:
        if "error" in res:
            # Keep error results separate
            continue

        key = (res["sample_id"], res["task"], res["library"])
        grouped[key].append(res)

    merged_results = []

    # Handle errors separately
    error_results = [r for r in results if "error" in r]
    merged_results.extend(error_results)

    # Process each task+library group
    for key, group_results in grouped.items():
        sample_id, task, library = key

        # Sort by chunk number
        group_results.sort(key=lambda x: x["chunk_num"])

        # Create a base record
        base_record = {
            "sample_id": sample_id,
            "task": task,
            "library": library,
            "sample_length": group_results[0]["sample_length"],
            "chunks_processed": len(group_results),
            "merged": True,
        }

        # Merge time metrics
        total_time = sum(r["time_taken"] for r in group_results)
        base_record["time_taken"] = total_time

        if "chars_per_second" in group_results[0]:
            # Calculate average processing speed
            speeds = [
                r["chars_per_second"] for r in group_results if "chars_per_second" in r
            ]
            base_record["chars_per_second"] = np.mean(speeds) if speeds else 0

        # Task-specific merging
        if task == "dialogue_detection":
            dialogues = []
            total_count = 0

            for r in group_results:
                if "count" in r and "sample_dialogues" in r:
                    total_count += r["count"]
                    for dialogue in r["sample_dialogues"]:
                        if dialogue not in dialogues:
                            dialogues.append(dialogue)

            # Remove potential duplicates from overlap
            base_record["count"] = total_count
            base_record["sample_dialogues"] = dialogues[:5]  # Take top 5 samples

        elif task == "ner":
            characters = []
            unique_count = 0

            for r in group_results:
                if "unique_count" in r and "characters" in r:
                    unique_count += r["unique_count"]
                    for char in r["characters"]:
                        if char not in characters:
                            characters.append(char)

            # Deduplicate characters across chunks
            base_record["unique_count"] = len(characters)
            base_record["characters"] = characters[:10]  # Take top 10

        elif task == "sentiment":
            sentences = []
            total_count = 0

            for r in group_results:
                if "emotional_count" in r and "emotional_sentences" in r:
                    total_count += r["emotional_count"]
                    for sent in r["emotional_sentences"]:
                        if sent not in sentences:
                            sentences.append(sent)

            base_record["emotional_count"] = total_count
            base_record["emotional_sentences"] = sentences[:5]  # Top 5 samples

        merged_results.append(base_record)

    return merged_results


def process_multiple_samples(samples: dict[str, str]) -> list[dict]:
    """
    Process multiple text samples in parallel using ProcessPoolExecutor.

    Args:
        samples: A dictionary mapping sample_id to sample_text.

    Returns:
        A list of aggregated results from all samples.  Returns an empty list
        if the input 'samples' is empty.
    """
    if not isinstance(samples, dict):
        raise TypeError("samples must be a dictionary")
    if not samples:
        logger.warning("No samples provided to process_multiple_samples.")
        return []

    # Start with memory cleanup
    logger.info("Starting fresh run - cleaning up memory...")
    clear_memory()

    logger.info(
        f"Using {MAX_WORKERS} parallel processes with chunk size of {CHUNK_SIZE} chars"
    )
    all_results = []

    # with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
    #     futures = {
    #         executor.submit(process_sample_in_chunks, sample_id, sample_text): sample_id
    #         for sample_id, sample_text in samples.items()
    #     }

    #     # Iterate through completed futures with a progress bar
    #     for future in tqdm(as_completed(futures), total=len(futures),
    #                       desc="Processing samples", unit="sample"):
    #         sample_id = futures[future]
    #         try:
    #             sample_result = future.result()  # Get result (or exception)
    #             if sample_result:
    #                 all_results.extend(sample_result)
    #                 logger.info(f"[DONE] Completed: {sample_id} ({len(sample_result)} results)")
    #             else:
    #                 logger.warning(f"✗ Completed (no results): {sample_id}") # Log empty
    #         except Exception as e:
    #             logger.error(f"✗ Error processing {sample_id}: {e}")
    #             # Consider adding more specific error handling here if needed

    for sample_id, sample_text in samples.items():
        try:
            sample_result = process_sample_in_chunks(sample_id, sample_text)
            if sample_result:
                all_results.extend(sample_result)
                logger.info(f"[DONE] Completed: {sample_id} ({len(sample_result)} results)")
            else:
                logger.warning(f"[DONE] Completed (no results): {sample_id}")
        except Exception as e:
            logger.error(f"[DONE] Error processing {sample_id}: {e}")

    # Final memory cleanup
    logger.info("Processing complete - cleaning up memory...")
    clear_memory()

    return all_results



def create_ground_truth(sample_id: str, text: str, dialogues: list[str] | None = None, characters: list[str] | None = None, emotions: list[tuple[str, float]] | None = None) -> dict:
    """Creates or updates ground truth data for a sample and saves it as JSON."""
    ground_truth = {
        "sample_id": sample_id,
        "text": text,
        "dialogues": dialogues or [],
        "characters": characters or [],
        "emotions": emotions or [],
    }
    filepath = os.path.join(GROUND_TRUTH_DIR, sample_id.replace("/", "_") + ".json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving ground truth for {sample_id}: {e}")


def load_ground_truth(sample_id: str) -> dict | None:
    """Loads ground truth data for a given sample ID."""
    filepath = os.path.join(GROUND_TRUTH_DIR, sample_id.replace("/", "_") + ".json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ground truth for {sample_id}: {e}")
            return None
    else:
        return None

# Cell 14: Analysis and Storage Function (Revised - from original Cell 13)

current_current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def analyze_and_save_results(all_results: list[dict], samples: dict[str, str], RESULTS_DIR: str, current_timestamp: str) -> pd.DataFrame:
    """
    Analyze benchmark results and save them to JSON and CSV files.

    Args:
        all_results: List of benchmark result dictionaries.
        samples: Dictionary of samples that were processed.
        RESULTS_DIR: Path to the results directory.
        current_timestamp: current_timestamp string for file naming.

    Returns:
        A pandas DataFrame containing the results.  Returns an empty DataFrame
        if 'all_results' is empty.
    """
    if not isinstance(all_results, list):
        raise TypeError("all_results must be a list")
    if not isinstance(samples, dict):
        raise TypeError("samples must be a dictionary")
    if not isinstance(RESULTS_DIR, str):  # Type check
        raise TypeError("RESULTS_DIR must be a string")
    if not isinstance(current_timestamp, str):  # Type check
        raise TypeError("current_timestamp must be a string")

    results_df = pd.DataFrame(all_results)
    logger.info("\n===== BENCHMARK RESULTS SUMMARY =====\n")

    if results_df.empty:
        logger.warning("No benchmark results available. Please check for errors.")
        return results_df

    # Create summary statistics
    try:
        summary = results_df.groupby(['task', 'library']).agg({
            'time_taken': ['mean', 'min', 'max', 'std'],
            'sample_length': ['mean', 'count']
        })
        logger.info(summary.to_string())  # Log the summary
    except KeyError as e:
        logger.error(f"KeyError during summary aggregation: {e}. Check result data format.")
        return pd.DataFrame() # Return an empty dataframe.
    except Exception as e:
        logger.error(f"Error during summary aggregation: {e}")
        return pd.DataFrame()


    # Calculate average performance by library and task
    logger.info("\n===== LIBRARY PERFORMANCE COMPARISON =====\n")
    try:
        library_task_performance = results_df.pivot_table(
            index='library',
            columns='task',
            values='time_taken',
            aggfunc='mean'
        )
        logger.info("Average processing time (seconds) by library and task:\n" + library_task_performance.to_string())

        library_performance = results_df.groupby('library')['time_taken'].mean()
        logger.info("\nOverall average processing time (seconds):\n" + library_performance.to_string())
    except KeyError as e:
        logger.error(f"KeyError during performance calculation: {e}.  Check result data format.")
        return results_df # Return what we have
    except Exception as e:
        logger.error(f"Error during performance calculations: {e}")
        return results_df # Return what we have

     # --- Save Results ---
    RESULTS_DIR_path = Path(RESULTS_DIR)  # Use pathlib
    RESULTS_DIR_path.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    result_file = RESULTS_DIR_path / f"benchmark_results_{current_timestamp}.json"
    try:
        with open(result_file, "w") as f:
            json.dump({
                "sample_count": len(samples),
                "current_timestamp": str(datetime.datetime.now()),
                "summary": {
                    "library_performance": library_performance.to_dict() if 'library_performance' in locals() else {},
                    "task_performance": library_task_performance.to_dict() if 'library_task_performance' in locals() else {}
                },
                "results": all_results
            }, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {result_file}")
    except Exception as e:
        logger.error(f"Error saving JSON results: {e}")

    # Create CSV for easier analysis
    csv_file = RESULTS_DIR_path / f"benchmark_results_{current_timestamp}.csv"
    try:
        results_df.to_csv(csv_file, index=False)
        logger.info(f"CSV results saved to {csv_file}")
    except Exception as e:
         logger.error(f"Error saving CSV results: {e}")

    return results_df

def create_visualizations(results_df: pd.DataFrame, RESULTS_DIR: str, current_timestamp: str) -> None:
    """
    Create and save visualizations of benchmark results.

    Args:
        results_df: DataFrame with benchmark results.
        RESULTS_DIR: Path object for the results directory.
        current_timestamp: current_timestamp string for file naming.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")
    if not isinstance(RESULTS_DIR, str): #Type Check
        raise TypeError("RESULTS_DIR must be a string")
    if not isinstance(current_timestamp, str):
        raise TypeError("current_timestamp must be a string")

    RESULTS_DIR_path = Path(RESULTS_DIR) # Use pathlib

    try:
        if not results_df.empty:
            # Set style
            sns.set(style="whitegrid")

            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Processing time by task and library (bar chart)
            time_by_task = results_df.pivot_table(
                index='task',
                columns='library',
                values='time_taken',
                aggfunc='mean'
            )
            time_by_task.plot(kind='bar', ax=axes[0, 0], title='Average Processing Time by Task')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_xlabel('Task')

            # 2. Processing speed comparison (bar chart)
            speed_data = results_df[results_df['task'] == 'processing_speed']
            if not speed_data.empty:
                sns.barplot(x='library', y='chars_per_second', data=speed_data, ax=axes[0, 1])
                axes[0, 1].set_title('Characters Processed Per Second')
                axes[0, 1].set_ylabel('Chars/second')
                axes[0, 1].set_xlabel('Library')

            # 3. NER performance (scatter plot)
            ner_data = results_df[results_df['task'] == 'ner']
            if not ner_data.empty:
                scatter = sns.scatterplot(
                    x='sample_length',
                    y='time_taken',
                    hue='library',
                    size='unique_count',
                    sizes=(20, 200),
                    data=ner_data,
                    ax=axes[1, 0]
                )
                axes[1, 0].set_title('NER Performance by Sample Size')
                axes[1, 0].set_xlabel('Sample Length (chars)')
                axes[1, 0].set_ylabel('Processing Time (seconds)')
                # Add legend title if needed
                if scatter.get_legend() is not None:
                    handles, labels = scatter.get_legend_handles_labels()
                    if len(handles) > 2:  # Make sure we have size-related handles
                        axes[1, 0].legend(handles=handles[2:], labels=labels[2:], title="Character Count")

            # 4. Time vs. text length for all tasks (line plot)
            sns.lineplot(
                x='sample_length',
                y='time_taken',
                hue='library',
                style='task',
                markers=True,
                data=results_df,
                ax=axes[1, 1]
            )
            axes[1, 1].set_title('Processing Time vs. Text Length')
            axes[1, 1].set_xlabel('Sample Length (chars)')
            axes[1, 1].set_ylabel('Time (seconds)')

            plt.tight_layout()

            # Save the figure
            plot_file = RESULTS_DIR_path / f"benchmark_plots_{current_timestamp}.png"
            plt.savefig(plot_file, dpi=300)
            logger.info(f"Visualization saved to {plot_file}")

            # --- Additional Visualizations ---

            # 1. Box plot of processing times
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='task', y='time_taken', hue='library', data=results_df)
            plt.title('Distribution of Processing Times by Task and Library')
            plt.ylabel('Time (seconds)')
            plt.tight_layout()
            box_plot_file = RESULTS_DIR_path / f"benchmark_boxplot_{current_timestamp}.png"
            plt.savefig(box_plot_file, dpi=300)
            logger.info(f"Box plot saved to {box_plot_file}")

            # 2. Heatmap of time correlation between tasks
            plt.figure(figsize=(10, 8))
            task_pivot = results_df.pivot_table(
                index=['sample_id', 'library'],
                columns='task',
                values='time_taken'
            )
            corr = task_pivot.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation of Processing Times Between Tasks')
            plt.tight_layout()
            corr_plot_file = RESULTS_DIR_path / f"benchmark_correlation_{current_timestamp}.png"
            plt.savefig(corr_plot_file, dpi=300)
            logger.info(f"Correlation plot saved to {corr_plot_file}")

            # 3. Direct comparison of libraries (scatter plot)
            plt.figure(figsize=(10, 6))
            comp_data = results_df.pivot_table(
                index='sample_id',
                columns='library',
                values='time_taken',
                aggfunc='sum'  # Sum times for all tasks for a sample
            )
            max_val = comp_data.max().max() if not comp_data.empty else 1  # Handle empty DataFrame
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal performance')

            if 'spacy' in comp_data.columns and 'nltk' in comp_data.columns:
                plt.scatter(comp_data['spacy'], comp_data['nltk'], alpha=0.7)
                plt.xlabel('spaCy Processing Time (seconds)')
                plt.ylabel('NLTK Processing Time (seconds)')
                plt.title('Direct Comparison: spaCy vs NLTK Performance')
                for idx, row in comp_data.iterrows():
                     if abs(row['spacy'] - row['nltk']) > max(row['spacy'], row['nltk']) * 0.5:
                        plt.text(row['spacy'], row['nltk'], idx.split('/')[-1], fontsize=8)
                plt.legend() # show the legend
                plt.tight_layout()
                comp_plot_file = RESULTS_DIR_path / f"library_comparison_{current_timestamp}.png"
                plt.savefig(comp_plot_file, dpi=300)
                logger.info(f"Library comparison plot saved to {comp_plot_file}")


            # Display all plots (optional, can be commented out for batch runs)
            plt.show()

        else:
            logger.warning("No data available for visualization. Check for errors in processing.")

    except Exception as e:
        logger.error(f"Could not create visualizations: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback

    logger.info("Benchmark visualization completed!")

def validate_against_ground_truth(results_df: pd.DataFrame, samples: dict[str, str], RESULTS_DIR: str, current_timestamp: str) -> pd.DataFrame:
    """
    Compare benchmark results against ground truth data.

    Args:
        results_df: DataFrame with benchmark results.
        samples: Dictionary of samples that were processed.
        RESULTS_DIR: Path to the results directory (string).
        current_timestamp: current_timestamp string for file naming.

    Returns:
        DataFrame with validation metrics (precision, recall, F1).
        Returns an empty DataFrame if no ground truth data is available.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame")
    if not isinstance(samples, dict):
        raise TypeError("samples must be a dictionary")
    if not isinstance(RESULTS_DIR, str): # Type check
        raise TypeError("RESULTS_DIR must be a string")
    if not isinstance(current_timestamp, str):  # Type check
        raise TypeError("current_timestamp must be a string")
    RESULTS_DIR_path = Path(RESULTS_DIR)

    validation_results = []

    for sample_id in samples.keys():
        gt = load_ground_truth(sample_id)
        if not gt:
            continue

        sample_results = results_df[results_df['sample_id'] == sample_id]

        for _, result in sample_results.iterrows():
            if result['task'] == 'ner' and 'characters' in result:
                gt_chars = set(gt.get('characters', []))
                detected_chars = set(result['characters'])

                true_positives = len(gt_chars.intersection(detected_chars))
                false_positives = len(detected_chars - gt_chars)
                false_negatives = len(gt_chars - detected_chars)

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                validation_results.append(
                    {
                        "sample_id": sample_id,
                        "task": "ner",
                        "library": result["library"],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }
                )

            # Dialogue Detection Validation
            if result["task"] == "dialogue_detection" and "sample_dialogues" in result:
                gt_dialogues = set(gt.get("dialogues", []))
                detected_dialogues = set(result["sample_dialogues"])
                true_positives = len(gt_dialogues.intersection(detected_dialogues))
                false_positives = len(detected_dialogues - gt_dialogues)
                false_negatives = len(gt_dialogues - detected_dialogues)

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                validation_results.append(
                    {
                        "sample_id": sample_id,
                        "task": "dialogue_detection",
                        "library": result["library"],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }
                )

            # Sentiment Validation (Example - Requires Ground Truth Format)
            if result["task"] == "sentiment" and "emotional_sentences" in result:
                gt_emotions = gt.get("emotions", [])  # List of (sentence, polarity)
                detected_emotions = result[
                    "emotional_sentences"
                ]  # List of (sentence, polarity)

                # Convert ground truth to set of sentences for easier comparison
                gt_sentences = set(sent for sent, _ in gt_emotions)

                true_positives = 0
                false_positives = 0
                for sent, _ in detected_emotions:  # Iterate through detected emotions
                    if sent in gt_sentences:  # compare sentences
                        true_positives += 1
                    else:
                        false_positives +=1
                false_negatives = len(gt_sentences) - true_positives

                precision = (
                        true_positives / (true_positives + false_positives)
                        if (true_positives + false_positives) > 0
                        else 0
                    )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                validation_results.append(
                    {
                        "sample_id": sample_id,
                        "task": "sentiment",
                        "library": result["library"],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                    }
                )

    # Convert to DataFrame
    validation_df = pd.DataFrame(validation_results)

    # Save validation results
    if not validation_df.empty:
        try:
            csv_filepath = RESULTS_DIR_path / f"validation_results_{current_timestamp}.csv"
            validation_df.to_csv(csv_filepath, index=False)
            logger.info(f"Validation results saved to {csv_filepath}")

            # Create validation visualization (F1 Score)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='library', y='f1', hue='task', data=validation_df)
            plt.title('F1 Score by Library and Task')
            plt.ylim(0, 1)
            plt.tight_layout()
            plot_filepath = RESULTS_DIR_path / f"validation_f1_{current_timestamp}.png"
            plt.savefig(plot_filepath, dpi=300)
            logger.info(f"Validation visualization saved to {plot_filepath}")
            plt.show()


        except Exception as e:
            logger.error(f"Error saving or visualizing validation results: {e}")

    else:
        logger.warning("No validation results to save or display.")

    return validation_df

if __name__ == "__main__":
    # --- Create dummy data and ground truth if no corpus exists ---
    corpus_path = Path(CORPUS_DIR)
    if not corpus_path.exists() or not any(
        corpus_path.glob("*.txt")
    ):  # more robust check
        logger.warning(f"No corpus found at {CORPUS_DIR}. Creating dummy data.")
        os.makedirs(CORPUS_DIR, exist_ok=True)
        os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

        dummy_samples = {
            "sample1.txt": 'This is a great movie! I absolutely loved it. "But the ending was so sad," she said.',
            "sample2.txt": "John Smith went to London. He met Jane Doe.  The food was terrible.",
        }

        for filename, text in dummy_samples.items():
            with open(os.path.join(CORPUS_DIR, filename), "w", encoding="utf-8") as f:
                f.write(text)

        # Create corresponding ground truth (adapt as needed)
        create_ground_truth(
            "category/sample1.txt",
            dummy_samples["sample1.txt"],
            dialogues=["But the ending was so sad,"],
            characters=[],
            emotions=[
                ("This is a great movie!", 0.8),
                ("I absolutely loved it.", 0.9),
                ("But the ending was so sad,", -0.7),
            ],
        )
        create_ground_truth(
            "category/sample2.txt",
            dummy_samples["sample2.txt"],
            dialogues=[],
            characters=["John Smith", "Jane Doe"],
            emotions=[("The food was terrible.", -0.75)],
        )

    # --- Load samples from disk ---
    samples = load_samples(directory=CORPUS_DIR)  # Use the configured directory

    # Display sample information
    logger.info(f"Loaded {len(samples)} samples:")
    for sample_id, sample_text in samples.items():
        logger.info(f"  - {sample_id} ({len(sample_text)} chars)")

    # --- Process the samples ---
    all_results = process_multiple_samples(samples)

    # --- Load ground truth ---
    all_ground_truth = {
        sample_id: load_ground_truth(sample_id) for sample_id in samples
    }

    # --- Analyze and save results ---
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Define current_timestamp
    results_df = analyze_and_save_results(all_results, samples, RESULTS_DIR, current_timestamp)
    # --- Validate Results ---
    if results_df is not None and not results_df.empty:
        validation_df = validate_against_ground_truth(results_df, samples)

        # --- Create Visualizations ---
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        create_visualizations(results_df, Path(RESULTS_DIR), current_timestamp)
    else:
        logger.warning(
            "No results to validate or visualize. Please check the previous steps."
        )

    # --- Cleanup (Moved from Cell 19) ---
    # remove_temp_files()  # Assuming you keep this function as defined previously
    # close_database_connections() # Keep this as well
    # release_resources() # Keep

    logger.info("Benchmark run complete. Cleanup performed.")


# Cell 18: Example Usage - Validation and Visualization (Revised - from original Cell 17)

# --- Validate Results ---
if results_df is not None and not results_df.empty:
    validation_df = validate_against_ground_truth(results_df, samples)

    # --- Create Visualizations ---
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    create_visualizations(results_df, Path(RESULTS_DIR), current_timestamp)
else:
    logger.warning(
        "No results to validate or visualize. Please check the previous steps."
    )


def remove_temp_files(directory: str = CORPUS_DIR) -> None:
    """
    Removes temporary files or directories created during the benchmark.

    Args:
        directory: The directory containing temporary files/directories.
                Defaults to the corpus directory.
    """
    if not isinstance(directory, str):
        raise TypeError("directory must be a string")

    temp_dir_path = Path(directory)
    if temp_dir_path.exists():
        try:
            # Check if it's the dummy data directory.  Be CAREFUL deleting directories.
            if (
                temp_dir_path.name == "corpus"
                and all(
                    child.name.startswith("sample")
                    for child in temp_dir_path.iterdir()
                    if child.is_file()
                )
                and all(
                    Path(
                        GROUND_TRUTH_DIR,
                        child.name.replace(".txt", ".json")
                        .replace("/", "_")
                        .replace("\\", "_"),
                    ).exists()
                    for child in temp_dir_path.iterdir()
                    if child.is_file()
                )
            ):
                # It *looks* like dummy data, but still be cautious.  Double-check paths.

                shutil.rmtree(temp_dir_path)
                logger.info(f"Removed temporary corpus directory: {temp_dir_path}")

                # Also remove the corresponding ground truth files.
                ground_truth_dir = Path(GROUND_TRUTH_DIR)
                for item in ground_truth_dir.iterdir():
                    if (
                        item.name.startswith("category_sample")
                        and item.suffix == ".json"
                    ):
                        item.unlink()
                logger.info(f"Removed corresponding temporary ground truth files.")

                # Remove empty directories.
                if not any(Path(GROUND_TRUTH_DIR).iterdir()):
                    try:
                        Path(GROUND_TRUTH_DIR).rmdir()
                        logger.info(f"Removed empty ground truth directory.")
                    except OSError as e:
                        logger.warning(f"Failed to remove ground truth directory: {e}")

            else:
                logger.warning(
                    f"Not removing directory {temp_dir_path}: Does not appear to be auto-generated dummy data."
                )

        except Exception as e:
            logger.error(f"Error removing temporary files/directories: {e}")
    else:
        logger.info(
            f"Temporary directory {temp_dir_path} does not exist. No cleanup needed."
        )


def close_database_connections() -> None:
    """
    Closes any open database connections.  Placeholder for database interaction.
    """
    # Placeholder for database connection closing.  Replace with your actual
    # database closing logic if you are using a database.
    logger.info("Closing database connections (placeholder).")
    # Example (if using SQLAlchemy):
    # if 'db_engine' in globals():
    #     db_engine.dispose()
    #     logger.info("Database connections closed.")


def release_resources() -> None:
    """
    Releases any other allocated resources (e.g., GPU memory).
    """
    # Placeholder for general resource release.
    logger.info("Releasing resources (placeholder).")
    # Example (if using PyTorch with CUDA):
    # try:
    #    import torch
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()
    #        logger.info("Cleared CUDA cache.")
    # except ImportError:
    #    logger.info("PyTorch not installed, skipping CUDA cache clearing.")
    clear_memory()  # We already have a function for this!


# Call the cleanup functions:
remove_temp_files()
close_database_connections()
release_resources()

logger.info("Benchmark run complete. Cleanup performed.")
