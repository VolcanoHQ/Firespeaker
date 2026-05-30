import os
import json
import time
import random # For jitter in backoff
import math # For exponential backoff calculation
from typing import List, Dict, Optional, Tuple # Added Optional, Tuple
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

# Import Groq and specific errors
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file. Please add it to your .env file."
    )

# --- Initialize Groq Client ---
try:
    # Increase default timeout slightly
    client = Groq(api_key=groq_api_key, timeout=30.0) # type: ignore
except Exception as e:
    messagebox.showerror("Groq Client Error", f"Failed to initialize Groq client: {e}")
    client = None

# --- Constants ---
CHUNK_SIZE = 15000
OVERLAP_SIZE = 1000
CONFIG_FILE = "grounded_truth_extractor_config.ini"
APP_TITLE = "Grounded Truth Extractor"

# --- Rate Limiting & Model Config ---
# Based on testing (~420 tokens/req), 6000 TPM / 420 tokens/req ≈ 14 req/min
# 60 sec / 14 req ≈ 4.3 sec/req. Add buffer.
DEFAULT_SLEEP_INTERVAL = 4.5  # Seconds between starting requests
MAX_API_RETRIES = 3 # Max retries for temporary errors (TPM/RPM limits, server errors)
INITIAL_BACKOFF_FACTOR = 1.0 # Seconds for first backoff
MAX_BACKOFF_TIME = 15.0 # Max wait time for exponential backoff

# Model priority list (best daily limits first)
# Ensure these model IDs are currently available in Groq's free tier
MODEL_PRIORITY_LIST = [
    "llama3-70b-8192", # Higher RPD/TPD
    "llama-3.3-70b-versatile", # Lower RPD/TPD
    "llama-3.3-70b-specdec", # Lower RPD/TPD
    # Add optional smaller models if needed as further fallbacks
    # "gemma2-9b-it", # Has good TPM/TPD
]
# Corresponding Daily Request Limits (RPD) - used to detect daily limit exhaustion
# Check Groq Limits page for current values! These are based on the previous table.
MODEL_RPD_LIMITS = {
    "llama3-70b-8192": 14400,
    "llama-3.3-70b-versatile": 1000,
    "llama-3.3-70b-specdec": 1000,
    "gemma2-9b-it": 14400, # Example value, check Groq docs
}
# Daily Token Limits (TPD) - harder to track, relying on RPD check primarily
MODEL_TPD_LIMITS = {
    "llama3-70b-8192": 500000,
    "llama-3.3-70b-versatile": 100000,
    "llama-3.3-70b-specdec": 100000,
    "gemma2-9b-it": 500000, # Example value, check Groq docs
}


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
        self.overwrite_chunks = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # --- Processing Control ---
        self.processing_active = False
        self.stop_requested = False
        self.processing_thread = None

        # --- Rate Limit & Model State ---
        self.current_model_index = 0 # Index in MODEL_PRIORITY_LIST
        self.rate_limit_sleep_duration = DEFAULT_SLEEP_INTERVAL # Base sleep
        self.last_api_call_end_time = time.monotonic() # Track end time for sleep calc

        # Store paths from the last successful processing for review
        self.last_processed_chunk_dir = None
        self.last_processed_original_text = None

        self.msg_queue = queue.Queue()
        self.load_config()
        self.create_widgets()
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Configuration Methods [UNCHANGED] ---
    def load_config(self):
        """Load saved configuration if it exists."""
        config = configparser.ConfigParser()
        config_path = Path(CONFIG_FILE)
        if config_path.exists():
            try:
                config.read(config_path)
                if 'Paths' in config:
                    self.input_path.set(config['Paths'].get('input', ''))
                    self.output_path.set(config['Paths'].get('output', ''))
                if 'Options' in config:
                    self.recursive.set(config['Options'].getboolean('recursive', False))
                    self.preserve_structure.set(config['Options'].getboolean('preserve_structure', False))
            except Exception as e:
                messagebox.showwarning("Config Load Error", f"Error loading configuration:\n{e}")

    def save_config(self):
        """Save current configuration."""
        config = configparser.ConfigParser()
        config['Paths'] = {'input': self.input_path.get(), 'output': self.output_path.get()}
        config['Options'] = {'recursive': str(self.recursive.get()), 'preserve_structure': str(self.preserve_structure.get())}
        try:
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            # Avoid logging during save if UI might be gone
            # self.log_message("Configuration saved")
        except Exception as e:
            # self.log_message(f"Error saving configuration: {e}") # Avoid logging
            print(f"Error saving configuration: {e}") # Print instead

    def on_closing(self):
        """Handle window close event."""
        if self.processing_active:
            if messagebox.askyesno("Exit Confirmation", "Processing is active. Are you sure you want to exit?"):
                self.stop_processing()
                # Allow some time for thread to potentially acknowledge stop
                self.root.after(200, self._shutdown)
            else:
                return # Don't close if user cancels
        else:
            self._shutdown()

    def _shutdown(self):
        """ Saves config and destroys window """
        self.save_config()
        self.root.destroy()


    # --- UI Creation & Handling [UNCHANGED] ---
    def create_widgets(self):
        """Create the UI widgets."""
        # --- Main frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        # --- Input section ---
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # --- Output section ---
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2, padx=5, pady=5)

        # --- Options section ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        options_frame.columnconfigure(0, weight=1)
        check_frame = ttk.Frame(options_frame)
        check_frame.grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(check_frame, text="Process subdirectories recursively", variable=self.recursive).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(check_frame, text="Preserve directory structure in output", variable=self.preserve_structure).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(options_frame, text="Save Settings", command=self.save_config).grid(row=0, column=1, sticky='e', padx=5, pady=5)

        # --- Process buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=15)
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing, width=15)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = ttk.Button(button_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.review_button = ttk.Button(button_frame, text="Review Chunks", command=self.select_and_open_chunk_review, width=15)
        self.review_button.pack(side=tk.LEFT, padx=10)

        # --- Status section ---
        status_frame = ttk.LabelFrame(main_frame, text="Status & Log", padding="10")
        status_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.rowconfigure(4, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_area = tk.Text(log_frame, wrap=tk.WORD, height=8, bd=0, highlightthickness=0)
        self.log_area.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_area.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_area.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)
        progress_label_frame = ttk.Frame(status_frame)
        progress_label_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(5,0))
        progress_label_frame.columnconfigure(0, weight=1)
        ttk.Label(progress_label_frame, textvariable=self.status_text).grid(row=0, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(progress_label_frame, variable=self.progress_value, length=200, mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0,5))

    def browse_input(self):
        """Open directory dialog to select input folder."""
        path = filedialog.askdirectory(title="Select Input Folder", initialdir=self.input_path.get() or Path.home())
        if path: self.input_path.set(path)

    def browse_output(self):
        """Open directory dialog to select output folder."""
        path = filedialog.askdirectory(title="Select Output Folder", initialdir=self.output_path.get() or Path.home())
        if path: self.output_path.set(path)

    # --- Threading & UI Update Methods [UNCHANGED] ---
    def log_message(self, message):
        """Add message to queue for logging in the text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.msg_queue.put(("log", f"[{timestamp}] {message}"))

    def update_status(self, message):
        """Update status message (also logs it)."""
        self.msg_queue.put(("status", message))
        self.log_message(f"Status: {message}")

    def update_progress(self, value):
        """Update progress bar value (0-100)."""
        self.msg_queue.put(("progress", value))

    def process_queue(self):
        """Process messages from the queue to update UI (runs in main thread)."""
        try:
            while True:
                msg_type, content = self.msg_queue.get_nowait()
                if msg_type == "log":
                    self.log_area.config(state=tk.NORMAL)
                    self.log_area.insert(tk.END, content + "\n")
                    self.log_area.see(tk.END)
                    self.log_area.config(state=tk.DISABLED)
                elif msg_type == "status": self.status_text.set(content)
                elif msg_type == "progress": self.progress_value.set(content)
                elif msg_type == "processing_finished": self.reset_ui()
                elif msg_type == "enable_stop": self.stop_button.config(state=tk.NORMAL)
                elif msg_type == "disable_start_stop":
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.DISABLED)
                self.msg_queue.task_done()
        except queue.Empty: pass
        finally: self.root.after(100, self.process_queue)

    # --- Processing Control Methods [UNCHANGED except reset_ui state] ---
    def start_processing(self):
        """Validate inputs and start processing in a separate thread."""
        if self.processing_active:
            messagebox.showwarning("Busy", "Processing is already active.")
            return
        if not client:
            messagebox.showerror("Groq Client Error", "Groq client is not initialized. Cannot start processing.")
            return
        input_p = self.input_path.get()
        output_p = self.output_path.get()
        if not input_p or not Path(input_p).is_dir():
            messagebox.showerror("Error", "Please select a valid input folder.")
            return
        if not output_p:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        try: Path(output_p).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output Error", f"Cannot create or access output folder:\n{output_p}\n{e}")
            return

        self.processing_active = True
        self.stop_requested = False
        self.current_model_index = 0 # Reset to primary model
        self.msg_queue.put(("disable_start_stop", None))
        self.msg_queue.put(("enable_stop", None))
        self.update_progress(0)
        self.log_message("="*20 + " Processing Started " + "="*20)

        self.processing_thread = threading.Thread(
            target=self.process_files_thread_target,
            args=(input_p, output_p, self.recursive.get(), self.preserve_structure.get()),
            daemon=True
        )
        self.processing_thread.start()

    def stop_processing(self):
        """Request the processing thread to stop."""
        if not self.processing_active: return
        self.log_message("Stop requested by user...")
        self.stop_requested = True
        self.stop_button.config(state=tk.DISABLED)

    def reset_ui(self):
        """Reset UI elements after processing finishes or stops."""
        self.processing_active = False
        # Don't reset stop_requested here, thread checks it
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED) # Always disable stop when not running
        self.progress_value.set(0)
        self.log_message("="*20 + " Processing Finished/Stopped " + "="*20)

    def process_files_thread_target(self, input_path_str, output_path_str, recursive, preserve_structure):
        """Target function for the processing thread, handles setup/teardown."""
        # Reset model index at the start of a full processing run
        self.current_model_index = 0
        try:
            self.process_files(input_path_str, output_path_str, recursive, preserve_structure)
        except Exception as e:
            self.log_message(f"FATAL PROCESSING ERROR: {e}")
            self.update_status(f"Error: {e}")
            import traceback
            self.log_message(traceback.format_exc()) # Log full traceback
        finally:
            self.msg_queue.put(("processing_finished", None))


    # --- Core Processing Logic ---

    def process_files(self, input_path_str, output_path_str, recursive, preserve_structure):
        """Process all text files in the input directory (runs in thread)."""
        self.update_status("Searching for text files...")
        input_path = Path(input_path_str); output_path = Path(output_path_str)
        glob_pattern = "**/*.txt" if recursive else "*.txt"
        files = sorted(list(input_path.glob(glob_pattern)))
        if not files: self.update_status("No text files found."); return
        num_files = len(files)
        self.log_message(f"Found {num_files} text files to process.")

        for i, file_path in enumerate(files):
            if self.stop_requested: self.update_status("Processing stopped by user."); return
            try:
                rel_path = file_path.relative_to(input_path)
                self.update_status(f"Processing: {rel_path} ({i+1}/{num_files})")
                if preserve_structure:
                    file_output_dir = output_path / rel_path.parent
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                else: file_output_dir = output_path
                self.process_file(file_path, file_output_dir) # Returns None or path string
                self.update_progress(((i + 1) / num_files) * 100)
            except Exception as e:
                self.log_message(f"ERROR processing {file_path.name}: {e}")
                import traceback
                self.log_message(traceback.format_exc()) # Log full traceback

        if not self.stop_requested:
            self.update_progress(100); self.update_status("Processing complete!")

    def process_file(self, file_path: Path, output_dir: Path) -> Optional[Path]: # Changed return type hint
        """
        Process a single text file: read, chunk, analyze, SAVE INDIVIDUAL CHUNKS.
        Returns the path to the chunk output directory on success, None otherwise.
        """
        filename = file_path.stem
        self.log_message(f"--- Starting file: {file_path.name} ---")
        try:
            text = self.read_text_file(file_path)
            original_text_path = output_dir / f"{filename}_original.txt"
            with open(original_text_path, "w", encoding="utf-8") as f: f.write(text)
        except Exception as e:
            self.log_message(f"  ERROR reading or saving original text for {file_path.name}: {e}")
            return None
        # text_snippet = text[:500] + "..." if len(text) > 500 else text # No longer needed for merged file

        self.log_message(f"  Splitting text (size: {CHUNK_SIZE}, overlap: {OVERLAP_SIZE})...")
        chunks = self.split_text_into_chunks(text, CHUNK_SIZE, OVERLAP_SIZE)
        num_chunks = len(chunks)
        self.log_message(f"  Split into {num_chunks} chunks.")
        if num_chunks == 0: self.log_message(f"  Skipping file {file_path.name} as it produced no chunks."); return None

        chunk_output_dir = output_dir / f"{filename}_chunks"; chunk_output_dir.mkdir(exist_ok=True)
        original_chunks_dir = output_dir / f"{filename}_original_chunks"; original_chunks_dir.mkdir(exist_ok=True)
        # Keep track if any chunk analysis fails fatally
        fatal_error_occurred = False
        processed_chunk_data = [] # Store successfully processed chunk data (optional, for unanalyzed text)


        for i in range(num_chunks):
            if self.stop_requested: self.log_message("  Processing stopped during chunk analysis."); return None
            chunk = chunks[i]; chunk_index = i + 1
            self.log_message(f"  Processing chunk {chunk_index}/{num_chunks}...")
            try:
                chunk_text_path = original_chunks_dir / f"chunk_{chunk_index}.txt"
                with open(chunk_text_path, "w", encoding="utf-8") as f: f.write(chunk)
            except Exception as e: self.log_message(f"    Warning: Could not save original text for chunk {chunk_index}: {e}")

            api_result = self.analyze_chunk_with_groq_robust(chunk, filename, chunk_index)

            # If analyze failed permanently after retries/fallbacks, stop processing this file
            if api_result is None or api_result.get("error") == "FATAL: All models rate limited or failed.":
                self.log_message(f"  FATAL ERROR analyzing chunk {chunk_index}. Stopping processing for file {filename}.")
                self.update_status(f"Error: API Failed on {filename} chunk {chunk_index}")
                fatal_error_occurred = True
                break # Exit the loop for this file

            # Save individual chunk JSON result (even if it contains non-fatal errors)
            if api_result is not None:
                processed_chunk_data.append(api_result) # Add for potential unanalyzed check
                try:
                    chunk_output_path = chunk_output_dir / f"chunk_{chunk_index}.json"
                    with open(chunk_output_path, "w", encoding="utf-8") as f:
                        json.dump(api_result, f, indent=2, ensure_ascii=False)
                except Exception as e: self.log_message(f"    Warning: Could not save JSON for chunk {chunk_index}: {e}")
            # No sleep here - handled within analyze_chunk_with_groq_robust

        # if not self.stop_requested:
        #     self.log_message("  Merging chunk results...")
        #     valid_chunk_data = [cd for cd in chunk_data_list if cd is not None] # Already filtered Nones
        #     if not valid_chunk_data: self.log_message("  No valid chunk data to merge."); return None
        #     final_json_data = self.merge_chunk_jsons(valid_chunk_data, filename, text_snippet) # Renamed merge function
        #     final_output_path = output_dir / f"{filename}.json"
        #     try:
        #         with open(final_output_path, "w", encoding="utf-8") as f: json.dump(final_json_data, f, indent=2, ensure_ascii=False)
        #         self.log_message(f"  Successfully saved final JSON: {final_output_path.name}")
        #         return str(final_output_path)
        #     except Exception as e: self.log_message(f"  ERROR saving final JSON {final_output_path.name}: {e}"); return None
        # else: self.log_message("  Merging skipped due to user stop request."); return None
        
        # --- New Logic: Check for Unanalyzed Chunks (Optional) & Return Chunk Dir Path ---
        if not self.stop_requested and not fatal_error_occurred:
            # Optional: Check for chunks with minimal data extracted (similar to before)
            unanalyzed_chunks_content = []
            for i, chunk_data in enumerate(processed_chunk_data):
                # Check if the chunk data exists and doesn't contain a non-fatal error key,
                # and lacks significant extracted content.
                if chunk_data and "error" not in chunk_data and \
                    not chunk_data.get("dialogues") and \
                    not chunk_data.get("emotions") and \
                    not chunk_data.get("characters"):
                        # We need the original chunk text here
                        if i < len(chunks): # Ensure index is valid
                            unanalyzed_chunks_content.append(chunks[i])

            if unanalyzed_chunks_content:
                try:
                    unanalyzed_path = output_dir / f"{filename}_potentially_unanalyzed.txt"
                    with open(unanalyzed_path, 'w', encoding='utf-8') as f:
                        f.write("\n\n===== POTENTIALLY UNANALYZED TEXT (Chunk Boundaries Approximate) =====\n\n".join(unanalyzed_chunks_content))
                    self.log_message(f"  Saved potentially unanalyzed text snippets to {unanalyzed_path.name}")
                except Exception as e:
                    self.log_message(f"  Warning: Could not save unanalyzed text snippets: {e}")

            # Log success for the file
            self.log_message(f"  Successfully processed all chunks for {filename}. Results in: {chunk_output_dir.name}")
            return chunk_output_dir # Return path to the directory containing chunk JSONs

        elif self.stop_requested:
                self.log_message(f"  Stopped processing chunks for {filename}.")
                return None
        else: # Fatal error occurred
                self.log_message(f"  Finished processing file {filename} with fatal errors.")
                return None


    def read_text_file(self, file_path: Path):
        """Read text from a file."""
        with open(file_path, "r", encoding="utf-8") as file: return file.read()


    def split_text_into_chunks(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """Split text into potentially overlapping chunks, respecting boundaries."""
        # [UNCHANGED - Assumed OK]
        chunks = []; start_pos = 0; text_len = len(text)
        while start_pos < text_len:
            end_pos = min(start_pos + chunk_size, text_len)
            if end_pos < text_len:
                para_break = text.rfind('\n\n', start_pos, end_pos)
                if para_break > start_pos + overlap_size: end_pos = para_break + 2
                else:
                    sent_break = -1
                    for punct in ['. ', '! ', '? ']: sent_break = max(sent_break, text.rfind(punct, start_pos, end_pos))
                    if sent_break > start_pos + overlap_size: end_pos = sent_break + 2
            chunk = text[start_pos:end_pos]; chunks.append(chunk)
            next_start = end_pos - overlap_size
            start_pos = max(next_start, start_pos + 1) if end_pos < text_len else text_len
        return chunks

    def analyze_chunk_with_groq_robust(
        self, chunk: str, filename: str, chunk_index: int
    ) -> Optional[Dict]:
        """
        Handles API call to Groq with rate limiting, retries, and model fallback.
        Returns the parsed JSON data on success, a dict with an error key on
        recoverable failure during analysis, or None on fatal error stopping the file.
        """
        if not client:
            self.log_message("    ERROR: Groq client not available. Skipping API call.")
            return {"error": "Groq client not initialized"}

        system_prompt = """You are an AI assistant specialized in analyzing literary texts... Output ONLY a valid JSON object...""" # Truncated for brevity
        user_prompt = f"""Analyze the following text chunk (chunk #{chunk_index} from file '{filename}')... TEXT CHUNK:\n{chunk}\n\nReturn ONLY the valid JSON object.""" # Truncated

        current_retry = 0
        backoff_time = INITIAL_BACKOFF_FACTOR

        # --- Rate Limit Delay ---
        # Simple strategy: Ensure minimum time has passed since last call ended
        current_time = time.monotonic()
        time_since_last = current_time - self.last_api_call_end_time
        wait_time = self.rate_limit_sleep_duration - time_since_last
        if wait_time > 0:
            self.log_message(f"    Rate limit sleep: {wait_time:.2f}s...")
            time.sleep(wait_time)
        # More complex: Add header-based prediction here if desired

        while True: # Loop handles model switching and retries
            if self.stop_requested: return None # Check for stop request

            if self.current_model_index >= len(MODEL_PRIORITY_LIST):
                self.log_message("    FATAL: All models in priority list have hit daily limits or failed.")
                self.update_status("Error: All API models unavailable.")
                # Set stop requested to prevent further processing?
                # self.stop_requested = True # Optional: Stop entire run
                return {"error": "FATAL: All models rate limited or failed."}

            model_to_use = MODEL_PRIORITY_LIST[self.current_model_index]
            self.log_message(f"    Attempting API call with model: {model_to_use} (Attempt {current_retry + 1}/{MAX_API_RETRIES+1})")

            try:
                start_time = time.monotonic()
                response = client.with_raw_response.chat.completions.create(
                    messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ],
                    model=model_to_use,
                    temperature=0.1,
                )
                completion = response.parse()
                headers = response.headers
                self.last_api_call_end_time = time.monotonic() # Update successful call time

                # Process successful response
                json_response = completion.choices[0].message.content.strip()
                # --- Clean and Parse JSON ---
                if json_response.startswith("```json"): json_response = json_response[7:].rstrip("` ")
                elif json_response.startswith("```"): json_response = json_response[3:].rstrip("` ")
                json_response = json_response.strip()
                if not json_response.startswith("{"): start_idx = json_response.find("{"); json_response = json_response[start_idx:] if start_idx != -1 else json_response
                if not json_response.endswith("}"): end_idx = json_response.rfind("}"); json_response = json_response[: end_idx + 1] if end_idx != -1 else json_response

                try:
                    parsed_data = json.loads(json_response)
                    # Optional validation
                    if not isinstance(parsed_data.get("dialogues"), list): parsed_data["dialogues"] = []
                    if not isinstance(parsed_data.get("characters"), list): parsed_data["characters"] = []
                    if not isinstance(parsed_data.get("emotions"), list): parsed_data["emotions"] = []

                    # Log success and token usage
                    token_usage = completion.usage.total_tokens if completion.usage else 'N/A'
                    self.log_message(f"    Success with {model_to_use}. Tokens Used: {token_usage}. Remaining RPD: {headers.get('x-ratelimit-remaining-requests', 'N/A')}")
                    # --- Potentially adjust sleep based on usage ---
                    # if isinstance(token_usage, int) and token_usage > 0:
                    #     # Very basic adjustment: more tokens -> slightly longer sleep next time
                    #     self.rate_limit_sleep_duration = min(max(DEFAULT_SLEEP_INTERVAL, token_usage / 100), 10.0) # Example adjustment
                    # ----------------------------------------------
                    return parsed_data # SUCCESS

                except json.JSONDecodeError as json_err:
                    self.log_message(f"    JSON parsing error with {model_to_use}: {json_err}")
                    # Treat as potentially retryable error or fail chunk
                    if current_retry < MAX_API_RETRIES:
                        self.log_message("     Retrying due to parse error...")
                        current_retry += 1
                        time.sleep(1.0 * (1.5**current_retry) + random.uniform(0.1, 0.5)) # Simple backoff
                        continue # Retry the loop
                    else:
                        return {"error": f"JSON parse failed after retries: {json_err}"}

            except RateLimitError as e:
                self.last_api_call_end_time = time.monotonic() # Update time even on failure
                error_headers = e.response.headers if hasattr(e, 'response') and e.response is not None else {}
                remaining_rpd_str = error_headers.get('x-ratelimit-remaining-requests', '-1') # Default to -1 if header missing
                retry_after_str = error_headers.get('retry-after')

                try: remaining_rpd = int(remaining_rpd_str)
                except ValueError: remaining_rpd = -1 # Treat non-integer as unknown

                self.log_message(f"    Rate Limit Error (429) with {model_to_use}. Remaining RPD header: {remaining_rpd_str}. Retry-After: {retry_after_str}")

                # --- Check if it's likely a DAILY limit for THIS model ---
                # Use a low threshold (e.g., <= 5) as the header might not be exactly 0 when limit is hit
                if 0 <= remaining_rpd <= 5:
                    self.log_message(f"    Suspected DAILY request limit hit for {model_to_use}. Switching to next model.")
                    self.current_model_index += 1
                    current_retry = 0 # Reset retry count for the new model
                    backoff_time = INITIAL_BACKOFF_FACTOR # Reset backoff
                    # Immediately continue to try the next model in the outer loop
                    continue
                else:
                    # --- Temporary Limit (TPM/RPM) ---
                    if current_retry < MAX_API_RETRIES:
                        wait = backoff_time
                        if retry_after_str:
                            try: wait = max(wait, float(retry_after_str)) # Use retry-after if provided and longer
                            except ValueError: pass # Ignore invalid retry-after
                        wait = min(wait, MAX_BACKOFF_TIME) # Cap wait time
                        jitter = random.uniform(0.1, 0.5)
                        self.log_message(f"    Temporary rate limit. Retrying in {wait + jitter:.2f}s...")
                        time.sleep(wait + jitter)
                        current_retry += 1
                        backoff_time *= 1.5 # Increase backoff for next potential temporary failure
                        continue # Retry the loop with the SAME model
                    else:
                        self.log_message("    Max retries reached for temporary rate limits.")
                        return {"error": "Temporary rate limit exceeded after retries"}

            except (APIConnectionError, APIStatusError) as e: # Handle other potential API errors
                self.last_api_call_end_time = time.monotonic()
                self.log_message(f"    API Error ({type(e).__name__}) with {model_to_use}: {e}")
                if current_retry < MAX_API_RETRIES:
                    wait = backoff_time
                    jitter = random.uniform(0.1, 0.5)
                    self.log_message(f"    Retrying API call after error in {wait + jitter:.2f}s...")
                    time.sleep(wait + jitter)
                    current_retry += 1
                    backoff_time *= 1.5
                    continue # Retry loop
                else:
                    self.log_message("    Max retries reached for API errors.")
                    return {"error": f"API error after retries: {e}"}

            except Exception as e: # Catch any other unexpected errors during analysis
                self.last_api_call_end_time = time.monotonic()
                self.log_message(f"    UNEXPECTED error during analysis with {model_to_use}: {e}")
                import traceback
                self.log_message(traceback.format_exc())
                # Decide if this is retryable or should fail the chunk
                return {"error": f"Unexpected analysis error: {e}"}

        # Should not be reached if logic is correct, but acts as a fallback
        return {"error": "Analysis loop completed without success or specific error"}


    # def merge_chunk_jsons(self, chunk_data_list: List[Dict], filename: str, text_snippet: str) -> Dict: # Renamed function
    #     """Merge multiple chunk JSONs into a single comprehensive JSON, handling errors."""
    #     # [UNCHANGED - Assumed OK]
    #     all_dialogues = []; all_characters = set(); all_emotions = []; errors = []
    #     dialogue_set = set(); emotion_text_set = set()
    #     for i, chunk_data in enumerate(chunk_data_list):
    #         if chunk_data.get("error"): errors.append(f"Chunk {i+1}: {chunk_data['error']}"); continue
    #         for dialogue in chunk_data.get("dialogues", []):
    #             if isinstance(dialogue, str) and dialogue.strip() and dialogue not in dialogue_set:
    #                 all_dialogues.append(dialogue); dialogue_set.add(dialogue)
    #         for character in chunk_data.get("characters", []):
    #             if isinstance(character, str) and character.strip(): all_characters.add(character)
    #         for emotion in chunk_data.get("emotions", []):
    #             if isinstance(emotion, list) and len(emotion) >= 1 and isinstance(emotion[0], str):
    #                 emotion_text = emotion[0].strip()
    #                 if emotion_text and emotion_text not in emotion_text_set:
    #                     score = 0.0
    #                     if len(emotion) >= 2:
    #                         try: score = float(emotion[1])
    #                         except (ValueError, TypeError): pass
    #                     all_emotions.append([emotion_text, score]); emotion_text_set.add(emotion_text)
    #     final_json = {
    #         "sample_id": filename, "text_snippet": text_snippet, "processing_errors": errors,
    #         "dialogues": all_dialogues, "characters": sorted(list(all_characters)),
    #         "emotions": all_emotions, "last_updated": datetime.now().isoformat()
    #     }
    #     return final_json


    # --- Review Window Methods [UNCHANGED] ---
    def open_original_text(self, text_path: Path):
        """Open the original text file in the default text editor."""
        if not isinstance(text_path, Path): text_path = Path(text_path)
        if not text_path.exists():
            messagebox.showwarning("File Not Found", f"Original text file not found:\n{text_path}", parent=self.root)
            return
        try:
            system = platform.system()
            if system == 'Windows': os.startfile(text_path)
            elif system == 'Darwin': subprocess.call(('open', str(text_path)))
            else: subprocess.call(('xdg-open', str(text_path)))
        except Exception as e: messagebox.showerror("Error", f"Could not open file '{text_path.name}':\n{e}", parent=self.root)

    def open_chunk_review_window(self, chunk_dir_path: Path, original_text_path: Path):
        """Open a new window for chunk review with corrected scrollbar placement."""
        # [UNCHANGED - Assumed OK, using the revised version from previous steps]
        review_window = tk.Toplevel(self.root); review_window.title(f"Chunk Review - {chunk_dir_path.name}"); review_window.geometry("1000x700"); review_window.minsize(800, 600); review_window.transient(self.root); review_window.grab_set()
        main_frame = ttk.Frame(review_window, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED); paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame = ttk.Frame(paned_window); paned_window.add(left_frame, weight=1)
        ttk.Label(left_frame, text="Chunks:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        listbox_frame = ttk.Frame(left_frame); listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        chunk_listbox = tk.Listbox(listbox_frame, width=30, exportselection=False); chunk_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chunk_scroll = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=chunk_listbox.yview); chunk_scroll.pack(side=tk.RIGHT, fill=tk.Y); chunk_listbox.config(yscrollcommand=chunk_scroll.set)
        right_frame = ttk.Frame(paned_window); paned_window.add(right_frame, weight=4)
        chunk_data = {}; chunk_files = []
        try:
            if not chunk_dir_path.is_dir(): messagebox.showerror("Error", f"Chunk directory not found:\n{chunk_dir_path}", parent=review_window); review_window.destroy(); return
            chunk_files = sorted(chunk_dir_path.glob("chunk_*.json"), key=lambda p: int(p.stem.split('_')[1]))
            if not chunk_files: messagebox.showwarning("Warning", f"No 'chunk_*.json' files found in:\n{chunk_dir_path}", parent=review_window)
            for chunk_file in chunk_files:
                try:
                    chunk_num = int(chunk_file.stem.split('_')[1]); chunk_listbox.insert(tk.END, f"Chunk {chunk_num}")
                    with open(chunk_file, "r", encoding="utf-8") as f: chunk_data[chunk_num] = json.load(f)
                except (ValueError, IndexError) as e: self.log_message(f"Warning: Skipping invalid chunk filename {chunk_file.name}: {e}"); messagebox.showwarning("File Skipping", f"Skipping invalid chunk filename:\n{chunk_file.name}", parent=review_window)
                except json.JSONDecodeError as json_e: chunk_data[chunk_num] = {"error": f"JSON Error: {str(json_e)}"}; self.log_message(f"Warning: Could not parse JSON in {chunk_file.name}: {json_e}")
                except Exception as e: chunk_data[chunk_num] = {"error": str(e)}; self.log_message(f"Warning: Could not load {chunk_file.name}: {e}")
        except Exception as e: messagebox.showerror("Error", f"Error accessing chunk directory:\n{chunk_dir_path}\n{str(e)}", parent=review_window); review_window.destroy(); return
        ttk.Button(left_frame, text="Open Full Original Text", command=lambda: self.open_original_text(original_text_path)).pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)
        notebook = ttk.Notebook(right_frame); notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        def create_text_tab(parent_notebook, tab_title):
            tab_frame = ttk.Frame(parent_notebook); parent_notebook.add(tab_frame, text=tab_title); text_area_frame = ttk.Frame(tab_frame); text_area_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            text_widget = tk.Text(text_area_frame, wrap=tk.WORD, bd=0, highlightthickness=0, relief=tk.FLAT, padx=2, pady=2); text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(text_area_frame, orient=tk.VERTICAL, command=text_widget.yview); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); text_widget.config(yscrollcommand=scrollbar.set); text_widget.config(state=tk.DISABLED); return text_widget
        original_text_widget = create_text_tab(notebook, "Original Chunk Text"); dialogues_text_widget = create_text_tab(notebook, "Dialogues"); characters_text_widget = create_text_tab(notebook, "Characters"); emotions_text_widget = create_text_tab(notebook, "Emotions")
        text_widgets = {"original": original_text_widget, "dialogues": dialogues_text_widget, "characters": characters_text_widget, "emotions": emotions_text_widget}
        chunk_text_dir = None
        try:
            base_name = chunk_dir_path.name
            if base_name.endswith("_chunks"): original_chunks_dir_name = f"{base_name.replace('_chunks', '')}_original_chunks"; chunk_text_dir = chunk_dir_path.parent / original_chunks_dir_name
            else: self.log_message("Warning: Could not reliably determine original chunks directory name.")
        except Exception as e: self.log_message(f"Warning: Error determining original chunks directory path: {e}")
        def display_chunk(event=None):
            selected_indices = chunk_listbox.curselection();
            if not selected_indices: return;
            selected_index = selected_indices[0]
            try: chunk_num_str = chunk_listbox.get(selected_index).split()[1]; chunk_num = int(chunk_num_str)
            except (IndexError, ValueError): self.log_message(f"Error: Could not parse chunk number from listbox item: {chunk_listbox.get(selected_index)}"); return
            for widget in text_widgets.values(): widget.config(state=tk.NORMAL); widget.delete(1.0, tk.END)
            original_content = "Original text directory not determined or not found."
            if chunk_text_dir and chunk_text_dir.is_dir():
                chunk_text_file = chunk_text_dir / f"chunk_{chunk_num}.txt"
                if chunk_text_file.exists():
                    try:
                        with open(chunk_text_file, "r", encoding="utf-8") as f: original_content = f.read()
                    except Exception as e: original_content = f"Error loading original text: {e}"
                else: original_content = f"Original text file not found:\n{chunk_text_file.name}"
            elif chunk_text_dir: original_content = f"Original text directory not found:\n{chunk_text_dir}"
            text_widgets["original"].insert(tk.END, original_content)
            if chunk_num in chunk_data:
                data = chunk_data[chunk_num]
                if data.get("error"): error_msg = f"Error loading/processing chunk {chunk_num}:\n{data['error']}"; [text_widgets[key].insert(tk.END, error_msg) for key in ["dialogues", "characters", "emotions"]]
                else:
                    dialogues = data.get("dialogues", []); text_widgets["dialogues"].insert(tk.END, ("\n\n".join(f"{i+1}. {d}" for i, d in enumerate(dialogues)) if dialogues else "No dialogues extracted."))
                    characters = data.get("characters", []); text_widgets["characters"].insert(tk.END, ("\n".join(f"{i+1}. {c}" for i, c in enumerate(characters)) if characters else "No characters extracted."))
                    emotions = data.get("emotions", [])
                    if emotions:
                        for i, emotion in enumerate(emotions):
                            if isinstance(emotion, list) and len(emotion) >= 1:
                                text = str(emotion[0]); score_str = "(score missing)"
                                if len(emotion) >= 2:
                                    try: score_str = f"({float(emotion[1]):.2f})"
                                    except: score_str = f"({emotion[1]})"
                                text_widgets["emotions"].insert(tk.END, f"{i+1}. {score_str} {text}\n\n")
                    else: text_widgets["emotions"].insert(tk.END, "No emotions extracted.")
            else: no_data_msg = f"No data loaded for Chunk {chunk_num}."; [text_widgets[key].insert(tk.END, no_data_msg) for key in ["dialogues", "characters", "emotions"]]
            for widget in text_widgets.values(): widget.config(state=tk.DISABLED)
        chunk_listbox.bind("<<ListboxSelect>>", display_chunk)
        if chunk_listbox.size() > 0: chunk_listbox.selection_set(0); display_chunk()
        else:
            for widget in text_widgets.values(): widget.config(state=tk.NORMAL); widget.insert(tk.END, "No chunk files were found to display."); widget.config(state=tk.DISABLED)
        review_window.wait_window()

    def select_and_open_chunk_review(self):
        """Ask user for chunk directory and open the review window."""
        # [UNCHANGED - Assumed OK]
        if self.processing_active: messagebox.showwarning("Busy", "Cannot open review window while processing is active.", parent=self.root); return
        initial_dir = self.output_path.get()
        if not initial_dir or not Path(initial_dir).is_dir(): initial_dir = (self.last_processed_chunk_dir.parent if self.last_processed_chunk_dir else Path.home())
        chunk_dir = filedialog.askdirectory(title="Select the '_chunks' folder for review", initialdir=initial_dir, parent=self.root)
        if not chunk_dir: return
        chunk_dir_path = Path(chunk_dir)
        if not chunk_dir_path.name.endswith("_chunks"): messagebox.showwarning("Invalid Selection", "Please select a directory whose name ends with '_chunks'.", parent=self.root); return
        original_text_path = None
        try:
            base_name = chunk_dir_path.name.replace("_chunks", ""); potential_original_path = chunk_dir_path.parent / f"{base_name}_original.txt"
            if potential_original_path.exists(): original_text_path = potential_original_path
        except Exception as e: self.log_message(f"Could not auto-detect original text path: {e}")
        if not original_text_path:
            original_txt = filedialog.askopenfilename(title=f"Select the corresponding FULL original text file (*_original.txt)", initialdir=chunk_dir_path.parent, filetypes=[("Original text files", "*_original.txt"), ("Text files", "*.txt"), ("All files", "*.*")], parent=self.root)
            if not original_txt: messagebox.showerror("Error", "Original text file selection cancelled or file not found.", parent=self.root); return
            original_text_path = Path(original_txt)
        self.open_chunk_review_window(chunk_dir_path, original_text_path)


# --- Main Execution ---
def main():
    """Main function to start the GUI application."""
    root = tk.Tk()
    try: # Set theme
        style = ttk.Style(root); available_themes = style.theme_names()
        if "clam" in available_themes: style.theme_use("clam")
        elif "vista" in available_themes: style.theme_use("vista")
        elif "aqua" in available_themes: style.theme_use("aqua")
    except Exception as e: print(f"Could not set theme: {e}")
    app = GroundedTruthExtractor(root)
    root.mainloop()

if __name__ == "__main__":
    main()