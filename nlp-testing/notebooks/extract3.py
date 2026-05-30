# -*- coding: utf-8 -*-
"""
GroundedTruthExtractor Application (using Anthropic Claude Haiku)
Author: Your Name/Company
Date: 2025-03-30
Description: Extracts structured data (dialogues, characters, emotions)
             from text files using Anthropic Claude API, saves results
             as individual JSON files per text chunk, and includes
             options for resuming/overwriting previous processing.
"""

# --- Required Installations ---
# pip install anthropic python-dotenv Pillow tk # Pillow might be needed by ttk themes, ensure tk is installed
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

# --- Use Anthropic Library ---
try:
    import anthropic

    # Specific exceptions can be imported if needed for finer control, e.g.:
    # from anthropic import APIError, RateLimitError, AnthropicError
except ImportError:
    messagebox.showerror(
        "Missing Dependency",
        "Required library 'anthropic' not found. Please install it (`pip install anthropic`).",
    )
    anthropic = None  # Set to None if import fails

from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
# --- Anthropic Configuration ---
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
# Define the model to use (using the standard Haiku ID)
ANTHROPIC_MODEL_ID = "claude-3-5-haiku-20241022"
# ANTHROPIC_MODEL_ID = "claude-3-5-haiku-20241022" # Or use the one from your example if you are sure it's correct/available

# --- *** ADDED DEBUGGING *** ---
print("-" * 20)
print("DEBUG: Loaded Anthropic Configuration:")
print(f"  Model ID: {ANTHROPIC_MODEL_ID}")
print(f"  API Key Loaded: {'Yes' if anthropic_api_key else 'NO!'}")
print("-" * 20)
# --- *** END DEBUGGING *** ---


# --- Check for required Anthropic variable ---
missing_vars = []
if not anthropic_api_key:
    missing_vars.append("ANTHROPIC_API_KEY")

if missing_vars:
    # Show error only if the library was imported, otherwise it's already handled
    if anthropic:
        try:
            root_temp = tk.Tk()
            root_temp.withdraw()  # Hide the root window
            messagebox.showerror(
                "Configuration Error",
                f"Missing required Anthropic environment variable: {', '.join(missing_vars)}. "
                "Please set ANTHROPIC_API_KEY in your .env file or environment and restart.",
            )
            root_temp.destroy()
        except (
            tk.TclError
        ):  # Handle cases where Tkinter isn't fully initialized or available
            print(
                f"ERROR: Missing required Anthropic environment variable: {', '.join(missing_vars)}"
            )
        # Regardless of messagebox, raise error to stop script execution if var missing
        raise ValueError(
            f"Missing required Anthropic environment variable: {', '.join(missing_vars)}"
        )


# --- Configure Anthropic Client ---
anthropic_client = None
anthropic_client_configured = False
if anthropic and not missing_vars:
    try:
        anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key,
            # Can add other options like timeout here if needed
            # max_retries=0 # We handle retries manually below
        )
        # Optional: Could add a simple test call here, e.g., a short completion
        anthropic_client_configured = True
        print(f"Anthropic Client Configured (Model: {ANTHROPIC_MODEL_ID}).")
    except Exception as e:
        messagebox.showerror(
            "Anthropic Error", f"Failed to configure Anthropic client: {e}"
        )
        print(
            f"ERROR: Failed to configure Anthropic client: {e}"
        )  # Also print to console
else:
    if (
        anthropic
    ):  # Only print this if the library exists but config failed/vars missing
        print("Anthropic prerequisites missing or configuration failed. API disabled.")

# --- Constants ---
CHUNK_SIZE = 10000  # Adjust based on model context window and typical text density
OVERLAP_SIZE = 500
CONFIG_FILE = "grounded_truth_extractor_config.ini"
APP_TITLE = "Grounded Truth Extractor (Anthropic Claude Haiku)"  # Updated Title

# --- Rate Limiting & Model Config ---
# Adjust these based on observed behavior or Anthropic's documented limits (if available)
DEFAULT_SLEEP_INTERVAL = 0.5  # Start with a conservative value
MAX_API_RETRIES = 4
INITIAL_BACKOFF_FACTOR = 1.5
MAX_BACKOFF_TIME = 60.0


# --- Main Application Class ---
class GroundedTruthExtractor:

    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title(APP_TITLE)  # Use updated title
        self.root.geometry("700x550")
        self.root.minsize(600, 500)
        self.root.resizable(True, True)

        # --- Variables ---
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.recursive = tk.BooleanVar(value=False)
        self.preserve_structure = tk.BooleanVar(value=False)
        self.overwrite_chunks = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # --- Processing Control ---
        self.processing_active = False
        self.stop_requested = False
        self.processing_thread = None

        # --- Rate Limit & Model State ---
        self.rate_limit_sleep_duration = DEFAULT_SLEEP_INTERVAL  # Base sleep
        self.last_api_call_end_time = time.monotonic()  # Track end time for sleep calc

        # Store Anthropic client configuration for easy access
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_model_id = ANTHROPIC_MODEL_ID
        self.anthropic_client = anthropic_client  # Store the initialized client
        self.anthropic_client_configured = (
            anthropic_client_configured  # Store configured status
        )

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
                    )
                # Anthropic config could be loaded here too, but .env is generally preferred for keys
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
            "overwrite_chunks": str(self.overwrite_chunks.get()),
        }
        try:
            with open(CONFIG_FILE, "w") as configfile:
                config.write(configfile)
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
        # Ensure thread is stopped if somehow still running
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_requested = True
            # Optionally wait a short time, but daemon=True should allow exit
            # self.processing_thread.join(timeout=0.5)
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
        ttk.Checkbutton(
            check_frame,
            text="Overwrite existing chunk outputs if found",
            variable=self.overwrite_chunks,
        ).pack(anchor=tk.W, padx=5, pady=2)
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
            while True:  # Process all messages currently in the queue
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
                    # Only enable if processing is still marked as active
                    if self.processing_active:
                        self.stop_button.config(state=tk.NORMAL)
                elif msg_type == "disable_start_stop":
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.DISABLED)
                self.msg_queue.task_done()
        except queue.Empty:
            pass  # No more messages
        except Exception as e:
            # Log unexpected queue processing errors
            print(f"Error processing message queue: {e}")
            try:  # Try logging to UI as well
                self.log_message(f"ERROR in UI queue processing: {e}")
            except:
                pass  # Avoid recursive errors if logging fails
        finally:
            # Schedule the next check
            self.root.after(100, self.process_queue)

    # --- Processing Control Methods ---
    def start_processing(self):
        """Validate inputs and start processing in a separate thread."""
        if self.processing_active:
            messagebox.showwarning(
                "Busy", "Processing is already active.", parent=self.root
            )
            return
        # --- Check Anthropic Client Configuration ---
        if not self.anthropic_client_configured or not self.anthropic_client:
            messagebox.showerror(
                "Anthropic Error",
                "Anthropic client not configured. Cannot start. Check ANTHROPIC_API_KEY and console logs.",
                parent=self.root,
            )
            # Print the debug info again
            print("-" * 20)
            print("DEBUG: Anthropic Configuration at Start Attempt:")
            print(f"  Model ID: {self.anthropic_model_id}")
            print(f"  API Key Loaded: {'Yes' if self.anthropic_api_key else 'NO!'}")
            print(f"  Client Object Exists: {'Yes' if self.anthropic_client else 'No'}")
            print(f"  Client Configured Flag: {self.anthropic_client_configured}")
            print("-" * 20)
            return
        # -------------------------------------

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
        self.msg_queue.put(("disable_start_stop", None))
        self.msg_queue.put(("enable_stop", None))  # Enable stop button
        self.update_progress(0)
        self.log_message(
            "=" * 20
            + " Processing Started (Using Anthropic Claude) "
            + "=" * 20  # Updated log
        )

        # --- Start processing thread ---
        if self.processing_thread and self.processing_thread.is_alive():
            self.log_message(
                "Warning: Previous processing thread still alive? Attempting to join..."
            )
            try:
                self.processing_thread.join(timeout=0.5)
            except Exception as e:
                self.log_message(f"Error joining old thread: {e}")

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
        # Disable stop button immediately to prevent multiple clicks
        self.stop_button.config(state=tk.DISABLED)

    def reset_ui(self):
        """Reset UI elements after processing finishes or stops."""
        was_stopped = self.stop_requested  # Capture state before resetting flags
        self.processing_active = False
        self.stop_requested = False  # Reset for next run
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(
            state=tk.DISABLED
        )  # Always disable stop when not running
        self.progress_value.set(0)

        # Log final status based on whether stop was requested *before* this reset
        if not was_stopped:
            self.log_message("=" * 20 + " Processing Finished " + "=" * 20)
        else:
            self.log_message("=" * 20 + " Processing Stopped " + "=" * 20)
            self.update_status("Stopped by user.")  # Set status explicitly

    def process_files_thread_target(
        self, input_path_str, output_path_str, recursive, preserve_structure
    ):
        """Target function for the processing thread, handles setup/teardown."""
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
        # [Mostly UNCHANGED - outer loop logic is independent of API]
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
            self.log_message("No text files found in the specified input directory.")
            return

        nf = len(files)
        self.log_message(f"Found {nf} files.")
        files_processed_count = 0
        encountered_fatal_error = False

        for i, fp in enumerate(files):
            if self.stop_requested:
                self.update_status("Stopping...")
                self.log_message("File processing loop stopped by user request.")
                break

            try:
                relp = fp.relative_to(ip)
                self.update_status(f"Processing: {relp} ({i+1}/{nf})")
                fod = op / relp.parent if pres else op
                fod.mkdir(parents=True, exist_ok=True)

                # --- Process file using the NEW analysis function ---
                chunk_output_path = self.process_file(fp, fod)  # Returns path or None
                # ---------------------------------------------------

                if chunk_output_path:
                    self.last_processed_chunk_dir = chunk_output_path
                    self.last_processed_original_text = fod / f"{fp.stem}_original.txt"
                    files_processed_count += 1
                elif self.stop_requested:
                    self.log_message(f"Stopped during processing of file: {fp.name}")
                    break
                else:
                    self.log_message(
                        f"Skipping progress update for {fp.name} due to internal errors."
                    )
                    encountered_fatal_error = True

                self.update_progress(((i + 1) / nf) * 100)

            except Exception as e:
                self.log_message(f"ERROR processing file {fp.name} (outer loop): {e}")
                self.log_message(traceback.format_exc())
                encountered_fatal_error = True

        # --- Final Status Update ---
        if not self.stop_requested:
            self.update_progress(100)
            if encountered_fatal_error:
                self.update_status(
                    f"Complete with errors ({files_processed_count}/{nf} files OK). Check log."
                )
            else:
                self.update_status(
                    f"Complete! ({files_processed_count}/{nf} files processed)"
                )

    def process_file(self, file_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single text file: read, chunk, analyze (Anthropic Claude), SAVE INDIVIDUAL CHUNKS.
        Handles resuming or overwriting based on self.overwrite_chunks.
        Returns the path to the chunk output directory on success/partial success, None otherwise.
        """
        filename = file_path.stem
        self.log_message(f"--- Starting file: {file_path.name} ---")

        # --- 1. Read & Save Original Text ---
        try:
            text_content = self.read_text_file(file_path)
            original_text_path = output_dir / f"{filename}_original.txt"
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
        try:
            text_chunks = self.split_text_into_chunks(
                text_content, CHUNK_SIZE, OVERLAP_SIZE
            )
        except Exception as e:
            self.log_message(f"  ERROR splitting text for {file_path.name}: {e}")
            return None

        num_chunks = len(text_chunks)
        self.log_message(f"  Split into {num_chunks} chunks.")
        if num_chunks == 0:
            self.log_message(f"  Skipping file {file_path.name}, no chunks produced.")
            return None

        # --- 3. Prepare Output Directories ---
        try:
            chunk_output_dir = output_dir / f"{filename}_chunks"
            original_chunks_dir = output_dir / f"{filename}_original_chunks"
            chunk_output_dir.mkdir(exist_ok=True)
            original_chunks_dir.mkdir(exist_ok=True)
        except Exception as e:
            self.log_message(f"  ERROR creating output directories for {filename}: {e}")
            return None

        # --- 4. Handle Resume / Overwrite ---
        start_chunk_index = 0
        chunk_data_list = ["<NOT_PROCESSED>"] * num_chunks  # Placeholder

        if self.overwrite_chunks.get():
            self.log_message(
                f"  Overwrite enabled. Clearing existing chunk outputs for {filename}..."
            )
            try:
                for json_file in chunk_output_dir.glob("chunk_*.json"):
                    json_file.unlink(missing_ok=True)
                for txt_file in original_chunks_dir.glob("chunk_*.txt"):
                    txt_file.unlink(missing_ok=True)
            except Exception as e:
                self.log_message(
                    f"  Warning: Could not clear all existing outputs in {chunk_output_dir.name}: {e}"
                )
        else:  # Attempt Resume
            existing_json_files = list(chunk_output_dir.glob("chunk_*.json"))
            if existing_json_files:
                highest_processed_chunk = 0
                valid_existing_chunks = 0
                self.log_message(
                    f"  Found existing chunk outputs. Loading previous results..."
                )
                for json_file in existing_json_files:
                    if self.stop_requested:
                        return None
                    try:
                        match = re.search(r"chunk_(\d+)\.json$", json_file.name)
                        if not match:
                            continue
                        chunk_num = int(match.group(1))
                        if 1 <= chunk_num <= num_chunks:
                            with open(json_file, "r", encoding="utf-8") as read_file:
                                chunk_data_list[chunk_num - 1] = json.load(read_file)
                            highest_processed_chunk = max(
                                highest_processed_chunk, chunk_num
                            )
                            valid_existing_chunks += 1
                        else:
                            self.log_message(
                                f"    Warning: Found chunk {chunk_num} outside current range (1-{num_chunks}). Deleting {json_file.name}."
                            )
                            json_file.unlink(missing_ok=True)
                    except (ValueError, IndexError) as e:
                        self.log_message(
                            f"    Warning: Error parsing chunk number/index for {json_file.name}: {e}. Deleting."
                        )
                        json_file.unlink(missing_ok=True)
                    except json.JSONDecodeError as e:
                        self.log_message(
                            f"    Warning: Skipping/deleting corrupt JSON {json_file.name}: {e}"
                        )
                        json_file.unlink(missing_ok=True)
                    except Exception as e:
                        self.log_message(
                            f"    Error processing existing file {json_file.name}: {e}"
                        )

                if highest_processed_chunk > 0:
                    start_chunk_index = highest_processed_chunk
                    self.log_message(
                        f"    Resuming processing from chunk {start_chunk_index + 1}. Loaded {valid_existing_chunks} previous results."
                    )
                else:
                    self.log_message(
                        "    No valid previous chunks found. Starting from beginning."
                    )

                # Verify/write original text for resumed chunks
                self.log_message(
                    "    Verifying/writing original text for resumed chunks..."
                )
                for i in range(start_chunk_index):
                    if self.stop_requested:
                        return None
                    if i < len(text_chunks):
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
        fatal_error_occurred_in_analysis = False
        for i in range(start_chunk_index, num_chunks):
            if self.stop_requested:
                self.log_message("  Processing stopped during chunk analysis.")
                return chunk_output_dir  # Return path even if stopped

            current_chunk_text = text_chunks[i]
            chunk_index = i + 1
            self.log_message(f"  Processing chunk {chunk_index}/{num_chunks}...")

            # Save original text for the current chunk
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

            # --- Call Anthropic Analysis ---
            api_result = self.analyze_chunk_with_anthropic_robust(  # <--- UPDATED CALL
                current_chunk_text, filename, chunk_index
            )
            # -----------------------------

            chunk_data_list[i] = api_result  # Store result (dict or None)

            is_fatal = False
            if api_result is None:
                self.log_message(
                    f"  Analysis stopped for chunk {chunk_index} (API call interrupted)."
                )
                return chunk_output_dir  # Stop file processing, return path
            elif isinstance(api_result, dict) and api_result.get(
                "error", ""
            ).startswith("FATAL:"):
                self.log_message(
                    f"  FATAL ERROR analyzing chunk {chunk_index}. Stopping processing for file {filename}."
                )
                self.update_status(
                    f"Error: API Failed on {filename} chunk {chunk_index}"
                )
                fatal_error_occurred_in_analysis = True
                is_fatal = True

            # Save individual chunk JSON result (if it's a dict)
            if isinstance(api_result, dict):
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

            if is_fatal:
                break  # Exit chunk loop for this file

        # --- 6. Post-processing and Return ---
        unanalyzed_indices = [
            i + 1 for i, data in enumerate(chunk_data_list) if data == "<NOT_PROCESSED>"
        ]
        if unanalyzed_indices:
            self.log_message(
                f"  Warning: Chunks {unanalyzed_indices} were not processed for {filename}."
            )

        if not fatal_error_occurred_in_analysis:
            self.log_message(
                f"  Successfully processed all chunks for {filename}. Results in: {chunk_output_dir.name}"
            )
        else:
            self.log_message(
                f"  Finished processing file {filename} with fatal errors during analysis."
            )

        return chunk_output_dir

    def read_text_file(self, file_path: Path):
        # [UNCHANGED from v2]
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            self.log_message(f"ERROR reading file {file_path}: {e}")
            raise

    # --- Updated split_text_into_chunks function ---
    def split_text_into_chunks(
        self, text: str, chunk_size: int, overlap_size: int
    ) -> List[str]:
        """Splits text into overlapping chunks, prioritizing paragraph and sentence boundaries."""
        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a string.")
        if chunk_size <= overlap_size:
            raise ValueError("Chunk size must be greater than overlap size.")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        chunks = []
        start_pos = 0
        text_len = len(text)
        # Define characters that mark the end of a sentence
        sentence_enders = (".", "!", "?")
        # Define quote characters to check after sentence enders
        quotes = ('"', "'", "”", "’")  # Include curly quotes

        while start_pos < text_len:
            # Determine potential end position based on chunk size
            end_pos = min(start_pos + chunk_size, text_len)
            actual_end_pos = end_pos  # Default to hard cut if no better break found

            # If not the last chunk, try to find a better break point before end_pos
            if end_pos < text_len:
                # 1. Prioritize paragraph breaks (\n\n)
                # Search backwards from the potential end position
                para_break = text.rfind("\n\n", start_pos, end_pos)
                # Ensure the break is found and is after the current start position
                if para_break != -1 and para_break > start_pos:
                    actual_end_pos = para_break + 2  # Include the double newline
                else:
                    # 2. If no suitable paragraph break, look for sentence breaks
                    # Search backwards from end_pos-1 down towards start_pos
                    best_sent_break = (
                        -1
                    )  # Store the position *after* the best break found
                    # Limit the backward search slightly to avoid tiny chunks if break is too early
                    search_limit = max(
                        start_pos, end_pos - chunk_size // 2
                    )  # Don't search back more than half the chunk size

                    for i in range(end_pos - 1, search_limit - 1, -1):
                        # Check if the character is a sentence ending punctuation
                        if text[i] in sentence_enders:
                            # Check if the punctuation is likely the end of a sentence
                            # Criteria: followed by space, newline, quote, or end of the search window
                            next_char_is_boundary = False
                            if i + 1 < end_pos:
                                # Check for space or newline
                                if text[i + 1].isspace():
                                    next_char_is_boundary = True
                                # Check for closing quote immediately after punctuation
                                elif text[i + 1] in quotes:
                                    # Optional: Check char after quote too? (e.g. ." \n) - adds complexity
                                    next_char_is_boundary = True
                            else:  # Punctuation is the last char in the potential chunk window
                                next_char_is_boundary = True

                            if next_char_is_boundary:
                                # Found a suitable sentence break point.
                                # Store the position *after* the punctuation.
                                best_sent_break = i + 1
                                # Stop searching backwards once the last suitable break is found
                                break

                    # Use the found sentence break if it's valid
                    if best_sent_break != -1:
                        actual_end_pos = best_sent_break
                    # else: No suitable paragraph or sentence break found,
                    # stick with actual_end_pos = end_pos (hard cut)

            # Extract the chunk based on the determined end position
            current_chunk_text = text[start_pos:actual_end_pos]
            if current_chunk_text:  # Avoid adding empty chunks
                chunks.append(current_chunk_text)

            # Calculate start of the next chunk
            # Move back by overlap size from the *actual* end position used
            next_start = actual_end_pos - overlap_size
            # Ensure we always move forward, even if overlap is large or chunk was short
            start_pos = max(next_start, start_pos + 1)

            # Safety break to prevent infinite loops in unexpected edge cases
            # Uses a simpler count check compared to the previous heuristic
            if (
                len(chunks) > text_len + 10
            ):  # Limit chunks to slightly more than text length
                self.log_message(
                    f"Warning: Excessive chunk count ({len(chunks)}) detected for text length {text_len}. Breaking split loop."
                )
                break

        return chunks

    # --- ANTHROPIC CLAUDE ANALYSIS FUNCTION ---
    def analyze_chunk_with_anthropic_robust(
        self, chunk: str, filename: str, chunk_index: int
    ) -> Optional[Dict]:
        """
        Handles API call to Anthropic Claude with rate limiting, retries.
        Uses the specified messages structure and expects JSON output.
        Returns a dictionary on success or error, None if stopped during call.
        """
        # Check if client is configured
        if not self.anthropic_client_configured or not self.anthropic_client:
            self.log_message("    ERROR: Anthropic client unavailable/unconfigured.")
            return {"error": "FATAL: Anthropic client unavailable"}

        # --- Define the Prompt using the user-provided structure ---
        # This is the detailed prompt structure you provided
        prompt_text_template = """You are tasked with analyzing a chunk of text to extract specific information about character dialogues, characters, and emotions. Your goal is to process the given text and provide a structured output in JSON format.

Here is the text chunk you need to analyze:

<text_chunk>
{text_chunk}
</text_chunk>

Follow these steps to complete the task:

1. Identify all character dialogue lines:
   - Look for text enclosed in quotation marks (" ") that represents spoken words by characters.
   - Include these dialogues exactly as they appear in the text, preserving punctuation within the quotes.

2. Identify all unique character names:
   - Look for names of characters mentioned in the text or speaking dialogue.
   - Include only distinct names, avoiding duplicates.

3. Identify text snippets expressing distinct emotions:
   - Look for phrases or sentences that convey emotions such as happiness, sadness, anger, fear, or surprise.
   - For each emotional snippet:
     a) Extract the relevant text.
     b) Assign a sentiment score as a float between -1.0 (very negative) and 1.0 (very positive).
     c) Consider the context and intensity of the emotion when assigning the score.

4. Format your output as a JSON object with the following structure:
   {{
     "dialogues": [list of dialogue strings],
     "characters": [list of unique character names],
     "emotions": [list of [emotional snippet, sentiment score] pairs]
   }}

5. Ensure that your output contains only the JSON object, with no additional text or explanations.

Remember to follow the JSON output format strictly. Double-check that your output is a valid JSON object containing only the requested information."""

        # Insert the actual chunk text into the template
        prompt_text = prompt_text_template.format(text_chunk=chunk)

        # Construct the messages payload for Anthropic
        messages_payload = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        # ---------------------------------------------

        current_retry = 0
        backoff_time = INITIAL_BACKOFF_FACTOR

        # --- Rate Limit Delay ---
        current_time = time.monotonic()
        time_since_last = current_time - self.last_api_call_end_time
        wait_time = self.rate_limit_sleep_duration - time_since_last
        if wait_time > 0:
            self.log_message(f"    Rate limit sleep: {wait_time:.2f}s...")
            sleep_end = time.monotonic() + wait_time
            while time.monotonic() < sleep_end:
                if self.stop_requested:
                    self.log_message("    Stop requested during rate limit sleep.")
                    return None
                time.sleep(0.1)

        # --- API Call and Retry Loop ---
        while current_retry <= MAX_API_RETRIES:
            if self.stop_requested:
                self.log_message("    Stop requested before API attempt.")
                return None

            model_name = self.anthropic_model_id
            self.log_message(
                f"    Attempting API call with {model_name} (Attempt {current_retry + 1}/{MAX_API_RETRIES+1})"
            )

            api_error = None  # Store caught error
            try:
                # --- ANTHROPIC API CALL ---
                start_time = time.monotonic()
                message = self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=8192,  # Max tokens for the *response*
                    temperature=0.0,  # Low temperature for deterministic JSON
                    messages=messages_payload,
                    # Add timeout if supported by the SDK version or handle externally
                    # timeout=120.0
                )
                # --------------------------
                api_call_duration = time.monotonic() - start_time
                self.last_api_call_end_time = time.monotonic()

                # --- Process and parse JSON response ---
                if (
                    not message.content
                    or not isinstance(message.content, list)
                    or not message.content[0].text
                ):
                    # Handle unexpected response structure
                    finish_reason = getattr(message, "stop_reason", "unknown")
                    self.log_message(
                        f"    Warning: Anthropic response missing expected content structure. Finish reason: {finish_reason}"
                    )
                    raise ValueError(
                        f"Anthropic response missing content. Finish reason: {finish_reason}"
                    )

                raw_response_text = message.content[0].text.strip()

                # Attempt to extract JSON robustly (similar to Azure version)
                json_response = None
                first_brace = raw_response_text.find("{")
                last_brace = raw_response_text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_response = raw_response_text[first_brace : last_brace + 1]
                    if first_brace > 0 or last_brace < len(raw_response_text) - 1:
                        self.log_message(
                            "    Warning: Extracted JSON, but found surrounding text in the Anthropic response."
                        )
                else:
                    self.log_message(
                        f"    Warning: Could not find JSON braces {{}} in Anthropic response. Attempting to parse raw."
                    )
                    self.log_message(
                        f"    Raw response snippet: {raw_response_text[:500]}..."
                    )
                    json_response = raw_response_text

                # --- Parse the extracted/raw JSON string ---
                try:
                    parsed_data = json.loads(json_response)

                    # Basic validation (same as before)
                    required_keys = {
                        "dialogues": list,
                        "characters": list,
                        "emotions": list,
                    }
                    for key, expected_type in required_keys.items():
                        if key not in parsed_data or not isinstance(
                            parsed_data.get(key), expected_type
                        ):
                            self.log_message(
                                f"    Warning: Key '{key}' missing or invalid type in JSON. Setting to empty list."
                            )
                            parsed_data[key] = expected_type()

                    # Deeper validation for emotions list
                    if isinstance(parsed_data.get("emotions"), list):
                        valid_emotions = []
                        for item in parsed_data["emotions"]:
                            if (
                                isinstance(item, list)
                                and len(item) == 2
                                and isinstance(item[0], str)
                            ):
                                try:
                                    _ = float(item[1])
                                    valid_emotions.append(item)
                                except (ValueError, TypeError):
                                    self.log_message(
                                        f"    Warning: Invalid score format: {item}. Skipping."
                                    )
                            else:
                                self.log_message(
                                    f"    Warning: Invalid emotion item format: {item}. Skipping."
                                )
                        if len(valid_emotions) != len(parsed_data["emotions"]):
                            self.log_message(f"    Corrected/filtered emotions list.")
                            parsed_data["emotions"] = valid_emotions

                    # Get usage data if available (structure might differ)
                    # Anthropic usage info is often in message.usage
                    usage_in = getattr(message.usage, "input_tokens", "N/A")
                    usage_out = getattr(message.usage, "output_tokens", "N/A")
                    self.log_message(
                        f"    Success with {model_name}. Tokens In: {usage_in}, Out: {usage_out}. Duration: {api_call_duration:.2f}s."
                    )
                    return parsed_data  # SUCCESS!

                except json.JSONDecodeError as json_err:
                    self.log_message(
                        f"    FATAL JSON parse error with {model_name}: {json_err}"
                    )
                    self.log_message(
                        f"    Response text (attempted parse): {json_response[:500]}..."
                    )
                    api_error = json_err  # Store error to trigger retry

            # --- Exception Handling for Anthropic API Calls ---
            # Use anthropic specific exceptions if available and imported
            except anthropic.RateLimitError as e:
                self.last_api_call_end_time = time.monotonic()
                self.log_message(
                    f"    Anthropic Rate Limit Error (429) with {model_name}: {e}. "
                )
                api_error = e
            except anthropic.APIConnectionError as e:
                self.last_api_call_end_time = time.monotonic()
                self.log_message(
                    f"    Anthropic API Connection Error with {model_name}: {e}"
                )
                api_error = e
            except anthropic.APIStatusError as e:  # Catches other non-2xx status codes
                self.last_api_call_end_time = time.monotonic()
                self.log_message(
                    f"    Anthropic API Status Error ({e.status_code}) with {model_name}: {e.message[:200]}..."
                )
                # Could add specific handling for 401 (auth), 403 (permission), 404 (model not found?)
                if e.status_code in [401, 403]:
                    self.log_message(
                        f"    FATAL: Received {e.status_code} error. Please check your ANTHROPIC_API_KEY and permissions."
                    )
                    return {
                        "error": f"FATAL: {e.status_code} Authentication/Permission Error."
                    }
                api_error = e
            except anthropic.AnthropicError as e:  # Catch broader Anthropic errors
                self.last_api_call_end_time = time.monotonic()
                self.log_message(
                    f"    Anthropic Library Error with {model_name}: {str(e)[:200]}..."
                )
                api_error = e
            except Exception as e:  # Catch other unexpected errors
                self.last_api_call_end_time = time.monotonic()
                self.log_message(
                    f"    Unexpected Error ({type(e).__name__}) during analysis: {str(e)[:200]}..."
                )
                api_error = e

            # --- Retry Logic ---
            if api_error:
                current_retry += 1
                if current_retry <= MAX_API_RETRIES:
                    # Anthropic SDK might handle Retry-After automatically if configured,
                    # but manual backoff provides more control and logging.
                    # Check if error object has retry_after info (less common than OpenAI?)
                    retry_after_seconds = getattr(
                        api_error, "retry_after", None
                    )  # Check if attribute exists

                    if (
                        retry_after_seconds
                        and isinstance(retry_after_seconds, (int, float))
                        and retry_after_seconds > 0
                    ):
                        wait = retry_after_seconds
                        self.log_message(
                            f"    Received retry_after suggestion: {wait}s."
                        )
                    else:
                        # Exponential backoff
                        wait = backoff_time * (1.5 ** (current_retry - 1))
                        wait = min(wait, MAX_BACKOFF_TIME)
                        wait += random.uniform(0.1, 0.9)  # Jitter

                    self.log_message(f"    Retrying in {wait:.2f}s...")
                    sleep_end = time.monotonic() + wait
                    while time.monotonic() < sleep_end:
                        if self.stop_requested:
                            self.log_message("    Stop requested during retry wait.")
                            return None
                        time.sleep(0.1)
                    # Continue to next retry attempt
                else:
                    # Max retries reached
                    self.log_message(
                        f"    FATAL: Max retries ({MAX_API_RETRIES}) reached for chunk {chunk_index}. Last error: {str(api_error)[:200]}"
                    )
                    return {
                        "error": f"FATAL: Max retries reached. Last error: {str(api_error)[:100]}"
                    }

        # --- End of Retry Loop ---
        self.log_message(
            f"    FATAL: Exceeded max retries ({MAX_API_RETRIES}) for chunk {chunk_index} without success."
        )
        return {"error": f"FATAL: Exceeded max retries ({MAX_API_RETRIES})"}

    # --- Review Window Methods [UNCHANGED - Logic is independent of API] ---
    def open_original_text(self, text_path: Path):
        # [Same as v2]
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
        # [Same as v2]
        if not chunk_dir_path or not chunk_dir_path.is_dir():
            messagebox.showerror(
                "Error",
                f"Cannot open review: Chunk directory invalid:\n{chunk_dir_path}",
                parent=self.root,
            )
            return
        if not original_text_path or not original_text_path.is_file():
            messagebox.showerror(
                "Error",
                f"Cannot open review: Original text file invalid:\n{original_text_path}",
                parent=self.root,
            )
            return
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
            chunk_files = sorted(
                [
                    p
                    for p in chunk_dir_path.glob("chunk_*.json")
                    if re.match(r"chunk_\d+\.json$", p.name)
                ],
                key=lambda p: int(re.search(r"chunk_(\d+)\.json$", p.name).group(1)),
            )
            if not chunk_files:
                messagebox.showwarning(
                    "Warning",
                    f"No valid 'chunk_*.json' files found in:\n{chunk_dir_path}",
                    parent=review_window,
                )
            for chunk_file in chunk_files:
                try:
                    chunk_num = int(
                        re.search(r"chunk_(\d+)\.json$", chunk_file.name).group(1)
                    )
                    chunk_listbox.insert(tk.END, f"Chunk {chunk_num}")
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        chunk_data[chunk_num] = json.load(f)
                except json.JSONDecodeError as json_e:
                    chunk_data[chunk_num] = {"error": f"JSON Error: {str(json_e)}"}
                    self.log_message(
                        f"Warning: Could not parse JSON in {chunk_file.name}: {json_e}"
                    )
                except Exception as e:
                    if "chunk_num" in locals():
                        chunk_data[chunk_num] = {"error": str(e)}
                    self.log_message(f"Warning: Could not load {chunk_file.name}: {e}")
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error accessing chunk directory contents:\n{chunk_dir_path}\n{str(e)}",
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
                state=tk.DISABLED,
            )
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(
                text_area_frame, orient=tk.VERTICAL, command=text_widget.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
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
            base_name_match = re.match(r"^(.*)_chunks$", chunk_dir_path.name)
            if base_name_match:
                original_chunks_dir_name = f"{base_name_match.group(1)}_original_chunks"
                potential_dir = chunk_dir_path.parent / original_chunks_dir_name
                if potential_dir.is_dir():
                    chunk_text_dir = potential_dir
                else:
                    self.log_message(
                        f"Warning: Original chunk directory not found: {potential_dir}"
                    )
            else:
                self.log_message(
                    f"Warning: Could not determine base name from {chunk_dir_path.name}"
                )
        except Exception as e:
            self.log_message(
                f"Warning: Error determining original chunks dir path: {e}"
            )

        def display_chunk(event=None):
            selected_indices = chunk_listbox.curselection()
            if not selected_indices:
                return
            selected_index = selected_indices[0]
            try:
                match = re.search(r"Chunk (\d+)", chunk_listbox.get(selected_index))
                if not match:
                    raise ValueError("Parse fail")
                chunk_num = int(match.group(1))
            except Exception as e:
                self.log_message(f"Error parsing chunk num from listbox: {e}")
                return
            for widget in text_widgets.values():
                widget.config(state=tk.NORMAL)
                widget.delete(1.0, tk.END)
            original_content = f"Original text for chunk {chunk_num} not found."
            if chunk_text_dir:
                chunk_text_file = chunk_text_dir / f"chunk_{chunk_num}.txt"
                if chunk_text_file.is_file():
                    try:
                        with open(chunk_text_file, "r", encoding="utf-8") as f:
                            original_content = f.read()
                    except Exception as e:
                        original_content = f"Error loading original text: {e}"
            else:
                original_content = (
                    "Original chunk text directory could not be determined."
                )
            text_widgets["original"].insert(tk.END, original_content)
            if chunk_num in chunk_data:
                data = chunk_data[chunk_num]
                if isinstance(data, dict) and data.get("error"):
                    error_msg = (
                        f"Error loading/processing chunk {chunk_num}:\n{data['error']}"
                    )
                    [
                        text_widgets[key].insert(tk.END, error_msg)
                        for key in ["dialogues", "characters", "emotions"]
                    ]
                elif isinstance(data, dict):
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
                        emotion_texts = []
                        for i, emotion in enumerate(emotions):
                            if isinstance(emotion, list) and len(emotion) >= 1:
                                text = str(emotion[0])
                                score_str = "(score missing/invalid)"
                                if len(emotion) >= 2:
                                    try:
                                        score_str = f"({float(emotion[1]):.2f})"
                                    except (ValueError, TypeError):
                                        score_str = f"({emotion[1]})"
                                emotion_texts.append(f"{i+1}. {score_str} {text}")
                            else:
                                emotion_texts.append(
                                    f"{i+1}. Invalid format: {emotion}"
                                )
                        text_widgets["emotions"].insert(
                            tk.END, "\n\n".join(emotion_texts)
                        )
                    else:
                        text_widgets["emotions"].insert(
                            tk.END, "No emotions extracted."
                        )
                else:
                    unexpected_data_msg = (
                        f"Unexpected data format for Chunk {chunk_num}: {type(data)}"
                    )
                    [
                        text_widgets[key].insert(tk.END, unexpected_data_msg)
                        for key in ["dialogues", "characters", "emotions"]
                    ]
            else:
                no_data_msg = f"No processed data found for Chunk {chunk_num}."
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
                widget.insert(tk.END, "No chunk files found or loaded.")
                widget.config(state=tk.DISABLED)
        review_window.wait_window()

    def select_and_open_chunk_review(self):
        # [Same as v2]
        if self.processing_active:
            messagebox.showwarning(
                "Busy",
                "Cannot open review window while processing is active.",
                parent=self.root,
            )
            return
        initial_dir_browse = self.output_path.get()
        if not initial_dir_browse or not Path(initial_dir_browse).is_dir():
            initial_dir_browse = (
                self.last_processed_chunk_dir.parent
                if self.last_processed_chunk_dir
                and self.last_processed_chunk_dir.parent.is_dir()
                else Path.home()
            )
        chunk_dir = filedialog.askdirectory(
            title="Select the '_chunks' folder for review",
            initialdir=initial_dir_browse,
            parent=self.root,
        )
        if not chunk_dir:
            return
        chunk_dir_path = Path(chunk_dir)
        if not chunk_dir_path.is_dir() or not chunk_dir_path.name.endswith("_chunks"):
            messagebox.showwarning(
                "Invalid Selection",
                f"Please select a valid directory ending with '_chunks'.\nSelected: {chunk_dir_path}",
                parent=self.root,
            )
            return
        original_text_path = None
        try:
            base_name_match = re.match(r"^(.*)_chunks$", chunk_dir_path.name)
            if base_name_match:
                base_name = base_name_match.group(1)
                potential_original_path = (
                    chunk_dir_path.parent / f"{base_name}_original.txt"
                )
                if potential_original_path.is_file():
                    original_text_path = potential_original_path
                else:
                    self.log_message(
                        f"Auto-detect failed: Original text not found at {potential_original_path}"
                    )
            else:
                self.log_message(
                    f"Could not determine base name from {chunk_dir_path.name}"
                )
        except Exception as e:
            self.log_message(f"Error during auto-detection of original text path: {e}")
        if not original_text_path:
            messagebox.showinfo(
                "Original Text Needed",
                "Could not automatically find the corresponding '_original.txt' file. Please select it manually.",
                parent=self.root,
            )
            original_txt = filedialog.askopenfilename(
                title=f"Select the FULL original text file for '{chunk_dir_path.name}'",
                initialdir=chunk_dir_path.parent,
                filetypes=[
                    ("Original text files", "*_original.txt"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
                parent=self.root,
            )
            if not original_txt:
                return
            original_text_path = Path(original_txt)
            if not original_text_path.is_file():
                messagebox.showerror(
                    "Error",
                    f"Selected original text path is not a valid file:\n{original_text_path}",
                    parent=self.root,
                )
                return
        self.open_chunk_review_window(chunk_dir_path, original_text_path)


# --- Main Execution ---
def main():
    """Main function to start the GUI application."""
    root = None
    try:
        # --- Check Anthropic library ---
        if not anthropic:
            return  # Error already shown

        # --- Check Anthropic Configuration ---
        if missing_vars:
            return  # Error already shown/raised

        # --- Check Client Initialization ---
        if not anthropic_client_configured or not anthropic_client:
            print("Exiting: Anthropic client could not be configured.")
            try:
                root_temp = tk.Tk()
                root_temp.withdraw()
                messagebox.showerror(
                    "Fatal Error",
                    "Anthropic client could not be configured. Check console logs and ANTHROPIC_API_KEY. Exiting.",
                )
                root_temp.destroy()
            except tk.TclError:
                pass
            return
        # ----------------------------------------------------

        root = tk.Tk()
        try:  # Apply theme
            style = ttk.Style(root)
            themes = style.theme_names()
            if "clam" in themes:
                style.theme_use("clam")
            elif "vista" in themes:
                style.theme_use("vista")
            elif "aqua" in themes:
                style.theme_use("aqua")
        except Exception as e:
            print(f"Theme Error: {e}")

        app = GroundedTruthExtractor(root)
        root.mainloop()

    except Exception as e:
        print(f"FATAL ERROR during application startup: {e}")
        print(traceback.format_exc())
        try:
            if root:
                root.destroy()
            root_temp = tk.Tk()
            root_temp.withdraw()
            messagebox.showerror(
                "Application Startup Error",
                f"A fatal error occurred during startup:\n\n{e}\n\nCheck console for details.",
            )
            root_temp.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
