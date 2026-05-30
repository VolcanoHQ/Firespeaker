# -*- coding: utf-8 -*-
"""
GroundedTruthExtractor Application (using Google AI SDK via Vertex AI)
Author: Your Name/Company
Date: 2025-03-30
Description: Extracts structured data (dialogues, characters, emotions)
            from text files using Google Gemini API, saves results
            as individual JSON files per text chunk, and includes
            options for resuming/overwriting previous processing.
"""

# --- Required Installations ---
# pip install google-generativeai python-dotenv Pillow # Pillow might be needed by ttk themes
import os
import json
import time
import random
import math
from typing import List, Dict, Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from pathlib import Path
import threading
import queue
import configparser
import platform
import subprocess
import re
import traceback

# --- Use Google AI Library ---
try:
    from google import genai
    from google.genai import types

    # Define HarmBlockThreshold for safety settings
    HarmBlockThreshold = types.HarmBlockThreshold
    # Import Google specific exceptions for potentially finer control later
    # from google.api_core import exceptions as google_exceptions
except ImportError:
    messagebox.showerror(
        "Missing Dependency", "Required library 'google-generativeai' not found."
    )
    genai = None
    types = None
    google_exceptions = None
    HarmBlockThreshold = None

from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION")  # e.g., "us-central1"

if not gcp_project_id or not gcp_location:
    if genai:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION required for Vertex AI."
        )

# --- Configure Google AI Client for Vertex AI ---
genai_client = None
genai_configured = False
if genai and gcp_project_id and gcp_location:
    try:
        genai_client = genai.Client(
            vertexai=True, project=gcp_project_id, location=gcp_location
        )
        genai_configured = True
        print(
            f"Google AI Client Configured (Vertex AI: {gcp_project_id}/{gcp_location})."
        )
    except Exception as e:
        messagebox.showerror("Google AI Error", f"Failed to configure client: {e}")
else:
    print("Google AI/Vertex prerequisites missing. API disabled.")

# --- Constants ---
CHUNK_SIZE = 15000
OVERLAP_SIZE = 1000
CONFIG_FILE = "grounded_truth_extractor_config.ini"
APP_TITLE = "Grounded Truth Extractor (Gemini on Vertex)"

# --- Rate Limiting & Model Config ---
DEFAULT_SLEEP_INTERVAL = 0.6  # Adjust based on actual Vertex quotas for flash-lite
MAX_API_RETRIES = 4
INITIAL_BACKOFF_FACTOR = 1.5
MAX_BACKOFF_TIME = 60.0

# --- Model Priority List (UPDATED) ---
MODEL_PRIORITY_LIST = [
    # New primary model based on sample
    "gemini-2.0-flash-lite-001",
    # Fallbacks from previous list (verify availability in your project/region)
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-1.0-pro-001",
]


# --- Main Application Class ---
class GroundedTruthExtractor:

    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("700x550")
        self.root.minsize(600, 500)
        self.root.resizable(True, True)

        # --- Variables ---
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.recursive = tk.BooleanVar(value=False)
        self.preserve_structure = tk.BooleanVar(value=False)
        self.overwrite_chunks = tk.BooleanVar(value=False)  # Resume/Overwrite option
        self.status_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # --- Processing Control ---
        self.processing_active = False
        self.stop_requested = False
        self.processing_thread = None

        # --- Rate Limit & Model State ---
        self.current_model_index = 0  # Index in MODEL_PRIORITY_LIST
        self.rate_limit_sleep_duration = DEFAULT_SLEEP_INTERVAL  # Base sleep
        self.last_api_call_end_time = time.monotonic()  # Track end time for sleep calc

        # Store paths from the last successful processing for review
        self.last_processed_chunk_dir = None
        self.last_processed_original_text = None

        self.msg_queue = queue.Queue()
        self.load_config()
        self.create_widgets()
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Configuration Methods ---
    def load_config(self):
        """Load saved configuration if it exists."""
        config = configparser.ConfigParser()
        config_path = Path(CONFIG_FILE)
        if config_path.exists():
            try:
                config.read(config_path)
                if "Paths" in config:
                    self.input_path.set(config["Paths"].get("input", ""))
                    self.output_path.set(config["Paths"].get("output", ""))
                if "Options" in config:
                    self.recursive.set(config["Options"].getboolean("recursive", False))
                    self.preserve_structure.set(
                        config["Options"].getboolean("preserve_structure", False)
                    )
                    self.overwrite_chunks.set(
                        config["Options"].getboolean("overwrite_chunks", False)
                    )  # Load overwrite setting
            except Exception as e:
                messagebox.showwarning(
                    "Config Load Error", f"Error loading configuration:\n{e}"
                )

    def save_config(self):
        """Save current configuration."""
        config = configparser.ConfigParser()
        config["Paths"] = {
            "input": self.input_path.get(),
            "output": self.output_path.get(),
        }
        config["Options"] = {
            "recursive": str(self.recursive.get()),
            "preserve_structure": str(self.preserve_structure.get()),
            "overwrite_chunks": str(
                self.overwrite_chunks.get()
            ),  # Save overwrite setting
        }
        try:
            with open(CONFIG_FILE, "w") as configfile:
                config.write(configfile)
            # Avoid logging during save if UI might be gone
        except Exception as e:
            print(f"Error saving configuration: {e}")  # Print instead of log

    def on_closing(self):
        """Handle window close event."""
        if self.processing_active:
            if messagebox.askyesno(
                "Exit Confirmation",
                "Processing is active. Are you sure you want to exit?",
            ):
                self.stop_processing()
                self.root.after(200, self._shutdown)  # Allow time for stop request
            else:
                return
        else:
            self._shutdown()

    def _shutdown(self):
        """Saves config and destroys window"""
        self.save_config()
        self.root.destroy()

    # --- UI Creation & Handling ---
    def create_widgets(self):
        """Create the UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Input Folder:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Output Folder:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=5
        )
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        options_frame.columnconfigure(0, weight=1)
        check_frame = ttk.Frame(options_frame)
        check_frame.grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            check_frame,
            text="Process subdirectories recursively",
            variable=self.recursive,
        ).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(
            check_frame,
            text="Preserve directory structure in output",
            variable=self.preserve_structure,
        ).pack(anchor=tk.W, padx=5, pady=2)
        # --- Add Overwrite Checkbutton ---
        ttk.Checkbutton(
            check_frame,
            text="Overwrite existing chunk outputs if found",
            variable=self.overwrite_chunks,
        ).pack(anchor=tk.W, padx=5, pady=2)
        # -------------------------------
        ttk.Button(options_frame, text="Save Settings", command=self.save_config).grid(
            row=0, column=1, sticky="e", padx=5, pady=5
        )

        # Process buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=15)
        self.start_button = ttk.Button(
            button_frame,
            text="Start Processing",
            command=self.start_processing,
            width=15,
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Processing",
            command=self.stop_processing,
            state=tk.DISABLED,
            width=15,
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.review_button = ttk.Button(
            button_frame,
            text="Review Chunks",
            command=self.select_and_open_chunk_review,
            width=15,
        )
        self.review_button.pack(side=tk.LEFT, padx=10)

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status & Log", padding="10")
        status_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.rowconfigure(4, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_area = tk.Text(
            log_frame, wrap=tk.WORD, height=8, bd=0, highlightthickness=0
        )
        self.log_area.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_area.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_area.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)
        progress_label_frame = ttk.Frame(status_frame)
        progress_label_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(5, 0))
        progress_label_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_label_frame, textvariable=self.status_text).grid(
            row=0, column=0, sticky="w"
        )
        self.progress_bar = ttk.Progressbar(
            progress_label_frame,
            variable=self.progress_value,
            length=200,
            mode="determinate",
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 5))

    def browse_input(self):
        path = filedialog.askdirectory(
            title="Select Input Folder", initialdir=self.input_path.get() or Path.home()
        )
        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=self.output_path.get() or Path.home(),
        )
        if path:
            self.output_path.set(path)

    # --- Threading & UI Update Methods ---
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.msg_queue.put(("log", f"[{timestamp}] {message}"))

    def update_status(self, message):
        self.msg_queue.put(("status", message))
        self.log_message(f"Status: {message}")

    def update_progress(self, value):
        self.msg_queue.put(("progress", value))

    def process_queue(self):
        try:
            while True:
                msg_type, content = self.msg_queue.get_nowait()
                if msg_type == "log":
                    self.log_area.config(state=tk.NORMAL)
                    self.log_area.insert(tk.END, content + "\n")
                    self.log_area.see(tk.END)
                    self.log_area.config(state=tk.DISABLED)
                elif msg_type == "status":
                    self.status_text.set(content)
                elif msg_type == "progress":
                    self.progress_value.set(content)
                elif msg_type == "processing_finished":
                    self.reset_ui()
                elif msg_type == "enable_stop":
                    self.stop_button.config(state=tk.NORMAL)
                elif msg_type == "disable_start_stop":
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.DISABLED)
                self.msg_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    # --- Processing Control Methods ---
    def start_processing(self):
        """Validate inputs and start processing in a separate thread."""
        if self.processing_active:
            messagebox.showwarning(
                "Busy", "Processing is already active.", parent=self.root
            )
            return
        if not genai or not genai_configured:
            messagebox.showerror(
                "Gemini Error",
                "Google AI client not configured. Cannot start.",
                parent=self.root,
            )
            return

        input_p = self.input_path.get()
        output_p = self.output_path.get()

        if not input_p or not Path(input_p).is_dir():
            messagebox.showerror(
                "Error", "Please select a valid input folder.", parent=self.root
            )
            return
        if not output_p:
            messagebox.showerror(
                "Error", "Please select an output folder.", parent=self.root
            )
            return
        try:
            Path(output_p).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror(
                "Output Error",
                f"Cannot create/access output folder:\n{output_p}\n{e}",
                parent=self.root,
            )
            return

        # --- Update UI state for processing ---
        self.processing_active = True
        self.stop_requested = False
        self.current_model_index = 0  # Reset to primary model
        self.msg_queue.put(("disable_start_stop", None))
        self.msg_queue.put(("enable_stop", None))
        self.update_progress(0)
        self.log_message(
            "=" * 20 + " Processing Started (Using Gemini/Vertex) " + "=" * 20
        )

        # --- Start processing thread ---
        self.processing_thread = threading.Thread(
            target=self.process_files_thread_target,  # Wrapper target
            args=(
                input_p,
                output_p,
                self.recursive.get(),
                self.preserve_structure.get(),
            ),
            daemon=True,  # Allows app to exit even if thread is running
        )
        self.processing_thread.start()

    def stop_processing(self):
        """Request the processing thread to stop."""
        if not self.processing_active:
            return
        self.log_message("Stop requested by user...")
        self.stop_requested = True
        self.stop_button.config(state=tk.DISABLED)  # Disable stop button after clicking

    def reset_ui(self):
        """Reset UI elements after processing finishes or stops."""
        self.processing_active = False
        # Don't reset stop_requested here, thread checks it
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(
            state=tk.DISABLED
        )  # Always disable stop when not running
        self.progress_value.set(0)
        # Add finished message only if processing wasn't stopped prematurely by user action causing shutdown
        if (
            not self.stop_requested
        ):  # Check if stop was requested before finishing normally
            self.log_message("=" * 20 + " Processing Finished " + "=" * 20)
        else:
            self.log_message("=" * 20 + " Processing Stopped " + "=" * 20)
        # Reset stop_requested flag *after* logging based on it
        # self.stop_requested = False # Or reset this only when starting a new process

    def process_files_thread_target(
        self, input_path_str, output_path_str, recursive, preserve_structure
    ):
        """Target function for the processing thread, handles setup/teardown."""
        self.current_model_index = 0  # Ensure reset at start of batch
        try:
            self.process_files(
                input_path_str, output_path_str, recursive, preserve_structure
            )
        except Exception as e:
            # Log any unexpected errors from the thread
            self.log_message(f"FATAL THREAD ERROR: {e}")
            self.log_message(traceback.format_exc())  # Log full traceback
            self.update_status(f"Fatal Error: {e}")
        finally:
            # Signal the main thread to reset the UI regardless of success/failure/stop
            self.msg_queue.put(("processing_finished", None))

    # --- Core Processing Logic ---
    def process_files(self, ips, ops, rec, pres):
        # [UNCHANGED from previous combined version]
        self.update_status("Searching...")
        ip = Path(ips)
        op = Path(ops)
        glob = "**/*.txt" if rec else "*.txt"
        try:
            files = sorted(list(ip.glob(glob)))
        except Exception as e:
            self.log_message(f"Error finding files: {e}")
            self.update_status("Error finding files.")
            return
        if not files:
            self.update_status("No text files found.")
            return
        nf = len(files)
        self.log_message(f"Found {nf} files.")
        for i, fp in enumerate(files):
            if self.stop_requested:
                self.update_status("Stopped.")
                return
            try:
                relp = fp.relative_to(ip)
                self.update_status(f"Processing: {relp} ({i+1}/{nf})")
                fod = op / relp.parent if pres else op
                fod.mkdir(parents=True, exist_ok=True)
                self.process_file(fp, fod)
                self.update_progress(((i + 1) / nf) * 100)
            except Exception as e:
                self.log_message(f"ERROR in file loop {fp.name}: {e}")
                self.log_message(traceback.format_exc())
        if not self.stop_requested:
            self.update_progress(100)
            self.update_status("Complete!")

    def process_file(self, file_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single text file: read, chunk, analyze (Gemini), SAVE INDIVIDUAL CHUNKS.
        Handles resuming or overwriting based on self.overwrite_chunks.
        Returns the path to the chunk output directory on success/partial success, None otherwise.

        Args:
            file_path (Path): Path to the input text file.
            output_dir (Path): Path to the directory where outputs for this file should be placed.

        Returns:
            Optional[Path]: Path to the directory containing chunk JSONs, or None if stopped/failed early.
        """
        filename = file_path.stem
        self.log_message(f"--- Starting file: {file_path.name} ---")

        # --- 1. Read & Save Original Text ---
        try:
            text_content = self.read_text_file(file_path)
            original_text_path = output_dir / f"{filename}_original.txt"
            # Save original text regardless of overwrite setting for chunks
            original_text_path.write_text(text_content, encoding="utf-8")
        except Exception as e:
            self.log_message(
                f"  ERROR reading or saving original text for {file_path.name}: {e}"
            )
            return None

        # --- 2. Split into Chunks ---
        self.log_message(
            f"  Splitting text (size: {CHUNK_SIZE}, overlap: {OVERLAP_SIZE})..."
        )
        text_chunks = self.split_text_into_chunks(
            text_content, CHUNK_SIZE, OVERLAP_SIZE
        )
        num_chunks = len(text_chunks)
        self.log_message(f"  Split into {num_chunks} chunks.")
        if num_chunks == 0:
            self.log_message(f"  Skipping file {file_path.name}, no chunks produced.")
            return None

        # --- 3. Prepare Output Directories ---
        chunk_output_dir = output_dir / f"{filename}_chunks"
        original_chunks_dir = output_dir / f"{filename}_original_chunks"
        chunk_output_dir.mkdir(exist_ok=True)
        original_chunks_dir.mkdir(exist_ok=True)

        # --- 4. Handle Resume / Overwrite ---
        start_chunk_index = 0  # Default: start from the beginning (index 0)
        chunk_data_list = [
            None
        ] * num_chunks  # Initialize list to hold results for all chunks

        if self.overwrite_chunks.get():
            # --- Overwrite Logic ---
            self.log_message(
                f"  Overwrite enabled. Clearing existing chunk outputs for {filename}..."
            )
            try:
                # Delete existing JSON chunks
                for json_file in chunk_output_dir.glob("chunk_*.json"):
                    json_file.unlink(
                        missing_ok=True
                    )  # Delete file, ignore if already gone
                # Delete existing original text chunks
                for txt_file in original_chunks_dir.glob("chunk_*.txt"):
                    txt_file.unlink(missing_ok=True)
            except Exception as e:
                self.log_message(
                    f"  Warning: Could not clear all existing outputs in {chunk_output_dir.name}: {e}"
                )
            # start_chunk_index remains 0
        else:
            # --- Attempt to Resume ---
            existing_json_files = list(chunk_output_dir.glob("chunk_*.json"))
            if existing_json_files:
                highest_processed_chunk = 0
                valid_existing_chunks = 0
                self.log_message(
                    f"  Found existing chunk outputs. Loading previous results..."
                )

                for json_file in existing_json_files:
                    try:
                        chunk_num_str = json_file.stem.split("_")[
                            -1
                        ]  # Get last part after underscore
                        chunk_num = int(chunk_num_str)

                        if (
                            1 <= chunk_num <= num_chunks
                        ):  # Check if chunk number is valid for current text
                            # Load JSON data into the corresponding list index
                            with open(json_file, "r", encoding="utf-8") as read_file:
                                chunk_data_list[chunk_num - 1] = json.load(read_file)
                            highest_processed_chunk = max(
                                highest_processed_chunk, chunk_num
                            )
                            valid_existing_chunks += 1
                        else:
                            # Chunk number from file doesn't match current split, delete it
                            self.log_message(
                                f"    Warning: Found chunk {chunk_num} which is outside the current range (1-{num_chunks}). Deleting {json_file.name}."
                            )
                            json_file.unlink(missing_ok=True)
                    except (ValueError, IndexError, json.JSONDecodeError) as e:
                        # Handle cases like non-integer chunk number or corrupt JSON
                        self.log_message(
                            f"    Warning: Skipping or deleting invalid/corrupt chunk file {json_file.name}: {e}"
                        )
                        json_file.unlink(missing_ok=True)  # Delete problematic file
                    except Exception as e:
                        # Catch other potential errors during file processing
                        self.log_message(
                            f"    Error processing existing file {json_file.name}: {e}"
                        )
                        # Decide whether to delete or just skip
                        # json_file.unlink(missing_ok=True)

                if highest_processed_chunk > 0:
                    # Set the starting index for the loop to the chunk *after* the last loaded one
                    start_chunk_index = highest_processed_chunk
                    self.log_message(
                        f"    Resuming processing from chunk {start_chunk_index + 1}. Loaded {valid_existing_chunks} previous results."
                    )
                else:
                    self.log_message(
                        "    No valid previous chunks found to resume from. Starting from beginning."
                    )

                # Ensure original text chunks exist for the chunks we just loaded (up to start_chunk_index)
                self.log_message(
                    "    Verifying/writing original text for resumed chunks..."
                )
                for i in range(
                    start_chunk_index
                ):  # Loop up to (but not including) start_chunk_index
                    if i < len(
                        text_chunks
                    ):  # Safety check index against actual chunks list
                        original_chunk_path = original_chunks_dir / f"chunk_{i+1}.txt"
                        if not original_chunk_path.exists():
                            try:
                                original_chunk_path.write_text(
                                    text_chunks[i], encoding="utf-8"
                                )
                            except Exception as e:
                                self.log_message(
                                    f"    Warning: Could not write missing original text for chunk {i+1}: {e}"
                                )
            else:
                self.log_message(
                    f"  No existing chunks found for {filename}. Starting fresh."
                )

        # --- 5. Process Remaining Chunks ---
        fatal_error_occurred = False
        # Create a working copy that includes any pre-loaded data
        processed_chunk_data_copy = chunk_data_list[:]

        # Loop from the determined start index to the end
        for i in range(start_chunk_index, num_chunks):
            if self.stop_requested:
                self.log_message("  Processing stopped during chunk analysis.")
                return None  # Stop requested

            current_chunk_text = text_chunks[i]
            chunk_index = i + 1  # Human-readable index (1-based)
            self.log_message(f"  Processing chunk {chunk_index}/{num_chunks}...")

            # Save original text for the chunk being processed (if overwriting or missing)
            try:
                current_original_chunk_path = (
                    original_chunks_dir / f"chunk_{chunk_index}.txt"
                )
                if (
                    not current_original_chunk_path.exists()
                    or self.overwrite_chunks.get()
                ):
                    current_original_chunk_path.write_text(
                        current_chunk_text, encoding="utf-8"
                    )
            except Exception as e:
                self.log_message(
                    f"    Warning: Could not save original text for chunk {chunk_index}: {e}"
                )

            # --- Call Gemini Analysis (robust version) ---
            api_result = self.analyze_chunk_with_gemini_robust(
                current_chunk_text, filename, chunk_index
            )
            # ---------------------------------------------

            # Store the result (or error dict, or None if stopped during analysis)
            chunk_data_list[i] = api_result
            processed_chunk_data_copy[i] = api_result  # Update working copy too

            # Check for fatal errors returned by analysis function
            if api_result is None or api_result.get("error", "").startswith("FATAL:"):
                self.log_message(
                    f"  FATAL ERROR analyzing chunk {chunk_index}. Stopping processing for file {filename}."
                )
                self.update_status(
                    f"Error: API Failed on {filename} chunk {chunk_index}"
                )
                fatal_error_occurred = True
                # Ensure error is stored if api_result was None
                if api_result is None:
                    chunk_data_list[i] = {
                        "error": "Fatal analysis error (None returned)"
                    }
                    processed_chunk_data_copy[i] = chunk_data_list[i]
                break  # Exit the loop for this file

            # Save individual chunk JSON result (even if it contains non-fatal errors)
            if api_result is not None:
                try:
                    chunk_json_path = chunk_output_dir / f"chunk_{chunk_index}.json"
                    with open(
                        chunk_json_path, "w", encoding="utf-8"
                    ) as json_write_file:
                        json.dump(
                            api_result, json_write_file, indent=2, ensure_ascii=False
                        )
                except Exception as e:
                    self.log_message(
                        f"    Warning: Could not save JSON for chunk {chunk_index}: {e}"
                    )

        # --- 6. Post-processing and Return ---
        if not self.stop_requested:
            # Optional: Check for potentially unanalyzed chunks using the final list
            unanalyzed_chunks_content = []
            for i, chunk_data in enumerate(
                processed_chunk_data_copy
            ):  # Use the copy reflecting the full run
                if (
                    chunk_data
                    and "error" not in chunk_data
                    and not chunk_data.get("dialogues")
                    and not chunk_data.get("emotions")
                    and not chunk_data.get("characters")
                ):
                    if i < len(text_chunks):  # Check index valid
                        unanalyzed_chunks_content.append(text_chunks[i])

            if unanalyzed_chunks_content:
                try:
                    unanalyzed_path = (
                        output_dir / f"{filename}_potentially_unanalyzed.txt"
                    )
                    unanalyzed_path.write_text(
                        "\n\n===== POTENTIALLY UNANALYZED TEXT =====\n\n".join(
                            unanalyzed_chunks_content
                        ),
                        encoding="utf-8",
                    )
                    self.log_message(f"  Saved potentially unanalyzed text snippets.")
                except Exception as e:
                    self.log_message(
                        f"  Warning: Could not save unanalyzed snippets: {e}"
                    )

            # Log final status for the file and return path to chunks
            if not fatal_error_occurred:
                self.log_message(
                    f"  Successfully processed all chunks for {filename}. Results in: {chunk_output_dir.name}"
                )
            else:
                self.log_message(
                    f"  Finished processing file {filename} with fatal errors."
                )
            return chunk_output_dir  # Return path even if fatal to allow review

        else:  # Stop was requested
            self.log_message(f"  Stopped processing {filename}.")
            return None

    def read_text_file(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_text_into_chunks(
        self, text: str, chunk_size: int, overlap_size: int
    ) -> List[str]:
        """
        Split text into potentially overlapping chunks, respecting boundaries.
        Corrected version to prevent UnboundLocalError.
        """
        chunks = []
        start_pos = 0
        text_len = len(text)

        while start_pos < text_len:
            # Determine potential end position
            end_pos = min(start_pos + chunk_size, text_len)

            # Initialize sent_break to -1 *before* checking conditions.
            # This ensures it always has a value when referenced later.
            sent_break = -1

            # Only try to adjust end_pos if we are not already at the very end of the text
            if end_pos < text_len:
                # --- Find Preferred Split Point ---
                # 1. Look for a paragraph break backward from the potential end_pos
                para_break = text.rfind("\n\n", start_pos, end_pos)

                # Check if the paragraph break is valid (not too close to the start, respecting overlap)
                if para_break > start_pos + overlap_size:
                    # Use paragraph break as the end position
                    end_pos = para_break + 2  # Include the newline characters
                    # In this case, sent_break remains -1, which is fine.

                else:
                    # 2. If no suitable paragraph break, look for a sentence break backward
                    # (sent_break was already initialized to -1 above)
                    for punct in [". ", "! ", "? "]:
                        current_find = text.rfind(punct, start_pos, end_pos)
                        sent_break = max(
                            sent_break, current_find
                        )  # Update if a later punctuation is found

                    # Check if the sentence break is valid (found and not too close to the start)
                    if sent_break > start_pos + overlap_size:
                        # Use sentence break as the end position
                        end_pos = sent_break + 2  # Include punctuation and space
                    # else: No suitable paragraph or sentence break found, use the original end_pos

            # --- Extract Chunk and Update Position ---
            current_chunk_text = text[start_pos:end_pos]
            chunks.append(current_chunk_text)

            # Calculate the start position for the next chunk, applying overlap
            next_start = end_pos - overlap_size

            # Ensure next_start doesn't go backward if the chunk was very short
            # or if overlap is large relative to the chunk size. Always advance at least one char.
            # Also handle reaching the end of the text.
            start_pos = (
                max(next_start, start_pos + 1) if end_pos < text_len else text_len
            )

        return chunks

    def analyze_chunk_with_gemini_robust(
        self, chunk: str, filename: str, chunk_index: int
    ) -> Optional[Dict]:
        """
        Handles API call to Gemini (Vertex) with rate limiting, retries, model fallback.
        Uses few-shot prompting and corrected config structure based on SDK examples.
        """
        if (
            not genai
            or not types
            or not HarmBlockThreshold
            or not genai_configured
            or not genai_client
        ):
            self.log_message(
                "    ERROR: Google AI client/types unavailable/unconfigured."
            )
            return {"error": "Google AI client unavailable"}

        # --- Define Generation Settings ---
        # Safety settings using correct SDK types and desired threshold
        # HarmBlockThreshold.BLOCK_NONE means disable blocking for that category
        current_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        # System instruction (optional - can also be part of the main prompt)
        # If used here, it MUST be a list containing a Part object
        # system_instruction_part = [types.Part.from_text("You are an assistant skilled at extracting specific data structures from literary text.")]

        # Create the GenerationConfig object
        current_gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",  # Crucial for requesting JSON output
            temperature=0.1,  # Low temperature for deterministic JSON
            max_output_tokens=8192,  # Max output size
            response_modalities=["TEXT"],  # Expect text modality
            # system_instruction=system_instruction_part, # Optional: Pass system instruction here
            safety_settings=current_safety_settings,  # Pass correctly formatted list here
        )
        # ----------------------------------------

        # --- Define the Prompt with Few-Shot Examples ---
        # Provide clear instructions and examples of the desired input/output format
        prompt = f"""**Task:** Analyze the provided text chunk and extract dialogues, characters, and emotions into a specific JSON format.

    **Instructions:**
    1. Identify all character dialogue.
    2. Identify all unique character names mentioned or speaking.
    3. Identify text snippets expressing distinct emotions and assign a sentiment score (float between -1.0 and 1.0).
    4. Output *only* a single, valid JSON object containing keys "dialogues" (list of strings), "characters" (list of strings), and "emotions" (list of ["snippet", score] lists). Do not include explanations, markdown, or any text outside the JSON braces.

    **Example 1:**

    Input Chunk:
    He slammed the door. "I cannot believe you did that!" John stared back, his face pale. "It wasn't my fault," he whispered, trembling slightly. A wave of sadness washed over Mary watching them.

    Output JSON only:
    {{
    "dialogues": [
        "I cannot believe you did that!",
        "It wasn't my fault,"
    ],
    "characters": [
        "John",
        "Mary"
    ],
    "emotions": [
        ["trembling slightly", -0.4],
        ["A wave of sadness washed over Mary", -0.7]
    ]
    }}

    **Example 2:**

    Input Chunk:
    Sunshine streamed through the window. Alice skipped down the path, humming happily. "What a wonderful day!" she exclaimed to a passing robin.

    Output JSON only:
    {{
    "dialogues": [
        "What a wonderful day!"
    ],
    "characters": [
        "Alice"
    ],
    "emotions": [
        ["humming happily", 0.8],
        ["What a wonderful day!", 0.9]
    ]
    }}

    **Input Chunk to Analyze (Chunk #{chunk_index} from '{filename}'):**

    {chunk}

    **Output JSON only:**
    """
        # ---------------------------------------------

        current_retry = 0
        backoff_time = INITIAL_BACKOFF_FACTOR
        # Rate Limit Delay
        current_time = time.monotonic()
        time_since_last = current_time - self.last_api_call_end_time
        wait_time = self.rate_limit_sleep_duration - time_since_last
        if wait_time > 0:
            self.log_message(f"    Rate limit sleep: {wait_time:.2f}s...")
            sleep_end = time.monotonic() + wait_time
            while time.monotonic() < sleep_end:
                if self.stop_requested:
                    return None
                time.sleep(0.1)

        while True:  # Retry/Fallback loop
            if self.stop_requested:
                return None
            if self.current_model_index >= len(MODEL_PRIORITY_LIST):
                self.log_message("    FATAL: All models failed.")
                return {"error": "FATAL: All models failed."}

            model_name = MODEL_PRIORITY_LIST[self.current_model_index]
            self.log_message(
                f"    Attempting API call with {model_name} (Attempt {current_retry + 1}/{MAX_API_RETRIES+1})"
            )

            try:
                # --- CORRECTED API CALL using 'config=' ---
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,  # The detailed prompt with examples
                    # Pass the GenerationConfig object to the 'config' parameter
                    config=current_gen_config,
                    # Safety settings are now *inside* the config object passed above
                    request_options={"timeout": 120},  # Increased timeout
                )
                # ---------------------------------------------
                self.last_api_call_end_time = time.monotonic()

                # Check response
                if not response.candidates:
                    feedback = getattr(response, "prompt_feedback", None)
                    reason = "Unknown"
                    if feedback:
                        try:
                            reason = feedback.block_reason.name
                        except AttributeError:
                            reason = str(getattr(feedback, "block_reason", "Unknown"))
                    raise ValueError(f"Response blocked/empty. Reason: {reason}")

                # Process and parse JSON
                json_response = response.text.strip()
                # Basic JSON finding (could be more robust if needed)
                first_brace = json_response.find("{")
                last_brace = json_response.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_response = json_response[first_brace : last_brace + 1]
                else:
                    raise json.JSONDecodeError(
                        "Valid JSON object structure not found", json_response, 0
                    )

                try:
                    # Parse the JSON
                    parsed_data = json.loads(json_response)

                    # Basic validation
                    for key, type_ in [
                        ("dialogues", list),
                        ("characters", list),
                        ("emotions", list),
                    ]:
                        if not isinstance(parsed_data.get(key), type_):
                            parsed_data[key] = type_()

                    # Get usage data safely
                    usage = "N/A"
                    try:
                        usage = response.usage_metadata.total_token_count
                    except Exception:
                        pass

                    self.log_message(f"    Success with {model_name}. Tokens: {usage}.")
                    return parsed_data  # SUCCESS!

                except json.JSONDecodeError as json_err:
                    self.log_message(
                        f"    JSON parse error with {model_name}: {json_err}"
                    )
                    self.log_message(f"    Snippet: {json_response[:300]}...")
                    raise  # Re-raise to trigger outer retry

            # --- Exception Handling ---
            except Exception as e:
                self.last_api_call_end_time = time.monotonic()
                err_name = type(e).__name__
                err_msg = str(e)
                is_rate_limit = (
                    "429" in err_msg or "Resource has been exhausted" in err_msg
                )
                is_permission = (
                    "403" in err_msg
                    or "PermissionDenied" in err_name
                    or "SERVICE_DISABLED" in err_msg
                )

                if is_permission:
                    self.log_message(
                        f"    Permission/Service Error ({err_name}) with {model_name}. Switching. Error: {err_msg[:200]}..."
                    )
                    self.current_model_index += 1
                    current_retry = 0
                    backoff_time = INITIAL_BACKOFF_FACTOR
                    time.sleep(0.5)
                    continue
                elif is_rate_limit:
                    self.log_message(f"    Rate Limit Error (429) with {model_name}.")
                    if current_retry < MAX_API_RETRIES:
                        wait = backoff_time
                        wait = min(wait, MAX_BACKOFF_TIME)
                        jitter = random.uniform(0.2, 0.8)
                        self.log_message(f"    Retrying in {wait + jitter:.2f}s...")
                        sleep_end = time.monotonic() + wait + jitter
                        while time.monotonic() < sleep_end:
                            if self.stop_requested:
                                return None
                            time.sleep(0.1)
                        current_retry += 1
                        backoff_time *= 1.8
                        continue
                    else:
                        self.log_message(
                            f"    Max retries for rate limits with {model_name}. Switching."
                        )
                        self.current_model_index += 1
                        current_retry = 0
                        backoff_time = INITIAL_BACKOFF_FACTOR
                        continue
                else:  # Other errors
                    self.log_message(
                        f"    API/Processing Error ({err_name}) with {model_name}: {err_msg[:200]}..."
                    )
                    if current_retry < MAX_API_RETRIES:
                        wait = backoff_time
                        wait = min(wait, MAX_BACKOFF_TIME)
                        jitter = random.uniform(0.2, 0.8)
                        self.log_message(
                            f"    Retrying after error in {wait + jitter:.2f}s..."
                        )
                        sleep_end = time.monotonic() + wait + jitter
                        while time.monotonic() < sleep_end:
                            if self.stop_requested:
                                return None
                            time.sleep(0.1)
                        current_retry += 1
                        backoff_time *= 1.8
                        continue
                    else:
                        self.log_message(
                            f"    Max retries for other errors with {model_name}. Switching."
                        )
                        self.current_model_index += 1
                        current_retry = 0
                        backoff_time = INITIAL_BACKOFF_FACTOR
                        continue

        # Only reached if loop exits without return
        self.log_message("    Analysis loop terminated without success")
        return {"error": "Analysis loop failed"}

    # --- Review Window Methods [UNCHANGED] ---
    def open_original_text(self, text_path: Path):
        # [Same as previous version]
        if not isinstance(text_path, Path):
            text_path = Path(text_path)
        if not text_path.exists():
            messagebox.showwarning(
                "File Not Found",
                f"Original text file not found:\n{text_path}",
                parent=self.root,
            )
            return
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(text_path)
            elif system == "Darwin":
                subprocess.call(("open", str(text_path)))
            else:
                subprocess.call(("xdg-open", str(text_path)))
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not open file '{text_path.name}':\n{e}",
                parent=self.root,
            )

    def open_chunk_review_window(self, chunk_dir_path: Path, original_text_path: Path):
        # [Same as previous version - Assumed OK]
        review_window = tk.Toplevel(self.root)
        review_window.title(f"Chunk Review - {chunk_dir_path.name}")
        review_window.geometry("1000x700")
        review_window.minsize(800, 600)
        review_window.transient(self.root)
        review_window.grab_set()
        main_frame = ttk.Frame(review_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        paned_window = ttk.PanedWindow(
            main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED
        )
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        ttk.Label(left_frame, text="Chunks:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        chunk_listbox = tk.Listbox(listbox_frame, width=30, exportselection=False)
        chunk_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chunk_scroll = ttk.Scrollbar(
            listbox_frame, orient=tk.VERTICAL, command=chunk_listbox.yview
        )
        chunk_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        chunk_listbox.config(yscrollcommand=chunk_scroll.set)
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=4)
        chunk_data = {}
        chunk_files = []
        try:
            if not chunk_dir_path.is_dir():
                messagebox.showerror(
                    "Error",
                    f"Chunk directory not found:\n{chunk_dir_path}",
                    parent=review_window,
                )
                review_window.destroy()
                return
            chunk_files = sorted(
                chunk_dir_path.glob("chunk_*.json"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if not chunk_files:
                messagebox.showwarning(
                    "Warning",
                    f"No 'chunk_*.json' files found in:\n{chunk_dir_path}",
                    parent=review_window,
                )
            for chunk_file in chunk_files:
                try:
                    chunk_num = int(chunk_file.stem.split("_")[1])
                    chunk_listbox.insert(tk.END, f"Chunk {chunk_num}")
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        chunk_data[chunk_num] = json.load(f)
                except (ValueError, IndexError) as e:
                    self.log_message(
                        f"Warning: Skipping invalid chunk filename {chunk_file.name}: {e}"
                    )
                    messagebox.showwarning(
                        "File Skipping",
                        f"Skipping invalid chunk filename:\n{chunk_file.name}",
                        parent=review_window,
                    )
                except json.JSONDecodeError as json_e:
                    chunk_data[chunk_num] = {"error": f"JSON Error: {str(json_e)}"}
                    self.log_message(
                        f"Warning: Could not parse JSON in {chunk_file.name}: {json_e}"
                    )
                except Exception as e:
                    chunk_data[chunk_num] = {"error": str(e)}
                    self.log_message(f"Warning: Could not load {chunk_file.name}: {e}")
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error accessing chunk directory:\n{chunk_dir_path}\n{str(e)}",
                parent=review_window,
            )
            review_window.destroy()
            return
        ttk.Button(
            left_frame,
            text="Open Full Original Text",
            command=lambda: self.open_original_text(original_text_path),
        ).pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        def create_text_tab(parent_notebook, tab_title):
            tab_frame = ttk.Frame(parent_notebook)
            parent_notebook.add(tab_frame, text=tab_title)
            text_area_frame = ttk.Frame(tab_frame)
            text_area_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            text_widget = tk.Text(
                text_area_frame,
                wrap=tk.WORD,
                bd=0,
                highlightthickness=0,
                relief=tk.FLAT,
                padx=2,
                pady=2,
            )
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(
                text_area_frame, orient=tk.VERTICAL, command=text_widget.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            text_widget.config(state=tk.DISABLED)
            return text_widget

        original_text_widget = create_text_tab(notebook, "Original Chunk Text")
        dialogues_text_widget = create_text_tab(notebook, "Dialogues")
        characters_text_widget = create_text_tab(notebook, "Characters")
        emotions_text_widget = create_text_tab(notebook, "Emotions")
        text_widgets = {
            "original": original_text_widget,
            "dialogues": dialogues_text_widget,
            "characters": characters_text_widget,
            "emotions": emotions_text_widget,
        }
        chunk_text_dir = None
        try:
            base_name = chunk_dir_path.name
            if base_name.endswith("_chunks"):
                original_chunks_dir_name = (
                    f"{base_name.replace('_chunks', '')}_original_chunks"
                )
                chunk_text_dir = chunk_dir_path.parent / original_chunks_dir_name
            else:
                self.log_message(
                    "Warning: Could not reliably determine original chunks directory name."
                )
        except Exception as e:
            self.log_message(
                f"Warning: Error determining original chunks directory path: {e}"
            )

        def display_chunk(event=None):
            selected_indices = chunk_listbox.curselection()
            if not selected_indices:
                return
            selected_index = selected_indices[0]
            try:
                chunk_num_str = chunk_listbox.get(selected_index).split()[1]
                chunk_num = int(chunk_num_str)
            except (IndexError, ValueError):
                self.log_message(
                    f"Error: Could not parse chunk number: {chunk_listbox.get(selected_index)}"
                )
                return
            for widget in text_widgets.values():
                widget.config(state=tk.NORMAL)
                widget.delete(1.0, tk.END)
            original_content = "Original text directory not determined or not found."
            if chunk_text_dir and chunk_text_dir.is_dir():
                chunk_text_file = chunk_text_dir / f"chunk_{chunk_num}.txt"
                if chunk_text_file.exists():
                    try:
                        with open(chunk_text_file, "r", encoding="utf-8") as f:
                            original_content = f.read()
                    except Exception as e:
                        original_content = f"Error loading original text: {e}"
                else:
                    original_content = (
                        f"Original text file not found:\n{chunk_text_file.name}"
                    )
            elif chunk_text_dir:
                original_content = (
                    f"Original text directory not found:\n{chunk_text_dir}"
                )
            text_widgets["original"].insert(tk.END, original_content)
            if chunk_num in chunk_data:
                data = chunk_data[chunk_num]
                if data.get("error"):
                    error_msg = (
                        f"Error loading/processing chunk {chunk_num}:\n{data['error']}"
                    )
                    [
                        text_widgets[key].insert(tk.END, error_msg)
                        for key in ["dialogues", "characters", "emotions"]
                    ]
                else:
                    dialogues = data.get("dialogues", [])
                    text_widgets["dialogues"].insert(
                        tk.END,
                        (
                            "\n\n".join(f"{i+1}. {d}" for i, d in enumerate(dialogues))
                            if dialogues
                            else "No dialogues extracted."
                        ),
                    )
                    characters = data.get("characters", [])
                    text_widgets["characters"].insert(
                        tk.END,
                        (
                            "\n".join(f"{i+1}. {c}" for i, c in enumerate(characters))
                            if characters
                            else "No characters extracted."
                        ),
                    )
                    emotions = data.get("emotions", [])
                    if emotions:
                        for i, emotion in enumerate(emotions):
                            if isinstance(emotion, list) and len(emotion) >= 1:
                                text = str(emotion[0])
                                score_str = "(score missing)"
                                if len(emotion) >= 2:
                                    try:
                                        score_str = f"({float(emotion[1]):.2f})"
                                    except:
                                        score_str = f"({emotion[1]})"
                                text_widgets["emotions"].insert(
                                    tk.END, f"{i+1}. {score_str} {text}\n\n"
                                )
                    else:
                        text_widgets["emotions"].insert(
                            tk.END, "No emotions extracted."
                        )
            else:
                no_data_msg = f"No data loaded for Chunk {chunk_num}."
                [
                    text_widgets[key].insert(tk.END, no_data_msg)
                    for key in ["dialogues", "characters", "emotions"]
                ]
            for widget in text_widgets.values():
                widget.config(state=tk.DISABLED)

        chunk_listbox.bind("<<ListboxSelect>>", display_chunk)
        if chunk_listbox.size() > 0:
            chunk_listbox.selection_set(0)
            display_chunk()
        else:
            for widget in text_widgets.values():
                widget.config(state=tk.NORMAL)
                widget.insert(tk.END, "No chunk files found.")
                widget.config(state=tk.DISABLED)
        review_window.wait_window()

    def select_and_open_chunk_review(self):
        # [Same as previous version]
        if self.processing_active:
            messagebox.showwarning(
                "Busy",
                "Cannot open review window while processing is active.",
                parent=self.root,
            )
            return
        initial_dir = self.output_path.get()
        if not initial_dir or not Path(initial_dir).is_dir():
            initial_dir = (
                self.last_processed_chunk_dir.parent
                if self.last_processed_chunk_dir
                else Path.home()
            )
        chunk_dir = filedialog.askdirectory(
            title="Select the '_chunks' folder for review",
            initialdir=initial_dir,
            parent=self.root,
        )
        if not chunk_dir:
            return
        chunk_dir_path = Path(chunk_dir)
        if not chunk_dir_path.name.endswith("_chunks"):
            messagebox.showwarning(
                "Invalid Selection",
                "Please select a directory ending with '_chunks'.",
                parent=self.root,
            )
            return
        original_text_path = None
        try:
            base_name = chunk_dir_path.name.replace("_chunks", "")
            potential_original_path = (
                chunk_dir_path.parent / f"{base_name}_original.txt"
            )
            if potential_original_path.exists():
                original_text_path = potential_original_path
        except Exception as e:
            self.log_message(f"Could not auto-detect original text path: {e}")
        if not original_text_path:
            original_txt = filedialog.askopenfilename(
                title=f"Select the corresponding FULL original text file (*_original.txt)",
                initialdir=chunk_dir_path.parent,
                filetypes=[
                    ("Original text files", "*_original.txt"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
                parent=self.root,
            )
            if not original_txt:
                messagebox.showerror(
                    "Error",
                    "Original text file selection cancelled or file not found.",
                    parent=self.root,
                )
                return
            original_text_path = Path(original_txt)
        self.open_chunk_review_window(chunk_dir_path, original_text_path)


# --- Main Execution ---
def main():
    """Main function to start the GUI application."""
    # Ensure genai and genai_configured are checked correctly
    if not genai:  # Correct indentation (typically 4 spaces)
        messagebox.showerror(
            "Fatal Error", "Required library 'google-genai' not found."
        )
        return  # Exit if library missing

    if not genai_configured:  # Correct indentation
        messagebox.showerror(
            "Fatal Error", "Google AI Client not configured. Check env vars/creds."
        )
        return  # Exit if client not configured

    root = tk.Tk()  # Correct indentation
    try:  # Correct indentation
        style = ttk.Style(root)  # Indented under try
        themes = style.theme_names()
        # Optional: print(f"Available themes: {themes}")
        if "clam" in themes:  # Indented under try
            style.theme_use("clam")
        elif "vista" in themes:  # Indented under try, aligned with if
            style.theme_use("vista")  # Windows fallback
        elif "aqua" in themes:  # Indented under try, aligned with if/elif
            style.theme_use("aqua")  # macOS fallback
    except Exception as e:  # Correct indentation, aligned with try
        print(f"Theme Error: {e}")  # Indented under except

    app = GroundedTruthExtractor(root)  # Correct indentation
    root.mainloop()  # Correct indentation


# This 'if' statement should be at the top level (no indentation)
if __name__ == "__main__":
    main()  # Call main() with standard indentation (typically 4 spaces) inside the 'if'
