import os
import time
import threading
import datetime
import uuid # To generate unique identifiers for prompts
from groq import Groq, RateLimitError
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("Error: GROQ_API_KEY not found in environment variables or .env file.")
    exit()

MODEL_A = "llama-3.3-70b-specdec"
MODEL_B = "llama-3.3-70b-versatile"
REQUESTS_TO_SEND = 4 # Keep the number of requests

BASE_PROMPT_TEXT = """
Please analyze the following short story excerpt and provide a concise summary (1-2 sentences),
identify the main characters, and list any explicitly mentioned emotions. Format the output as JSON.
Include the unique identifier provided in your response structure.

Excerpt:
The old lighthouse keeper, Silas, squinted against the relentless gale. Rain lashed the glass panes of the lantern room, mirroring the turmoil in his heart. For weeks, a gnawing anxiety had kept sleep at bay. His daughter, Elara, was sailing home through this very storm, her small fishing boat no match for the sea's fury. He gripped the brass railing, knuckles white, sending a silent prayer into the howling wind. Down below, the waves crashed against the rocks with a thunderous roar, each impact echoing his fear. Suddenly, a faint light flickered on the horizon. Hope surged, battling the dread that had settled deep within him. Was it her? Or just another cruel trick of the storm?

Unique Identifier: {unique_id}
"""

# --- Groq Client ---
try:
    client = Groq(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    exit()

# --- Shared Lock for Printing ---
print_lock = threading.Lock()

# --- Function to Make API Call and Print Headers/Usage ---
def make_request(model_id, request_num):
    """Makes a single API request, prints headers and token usage."""
    thread_id = threading.current_thread().name
    unique_id = str(uuid.uuid4())
    prompt_with_id = BASE_PROMPT_TEXT.format(unique_id=unique_id)

    with print_lock:
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {thread_id}: Sending request {request_num} using {model_id} (ID: ...{unique_id[-6:]})")

    start_time = time.monotonic()
    headers = {}
    status = "UNKNOWN"
    error_info = ""
    retry_after = "N/A"
    consumed_tokens = "N/A" # Initialize consumed tokens

    try:
        response = client.with_raw_response.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON formatting assistant."},
                {"role": "user", "content": prompt_with_id}
            ],
            model=model_id,
            temperature=0.2,
            max_tokens=350
        )
        completion = response.parse()
        headers = response.headers
        status = "SUCCESS"
        # --- Extract token usage ---
        if completion.usage:
            consumed_tokens = completion.usage.total_tokens
        else:
            consumed_tokens = "N/A (usage field missing)"
        # ---------------------------

    except RateLimitError as e:
        status = "FAILED (429 Rate Limit)"
        error_info = str(e)
        consumed_tokens = "N/A (Rate Limited)"
        if hasattr(e, 'response') and e.response is not None:
            headers = e.response.headers
            retry_after = headers.get('retry-after', 'N/A')
        else:
            headers = {}

    except Exception as e:
        status = "FAILED (Other Error)"
        error_info = str(e)
        consumed_tokens = f"N/A (Error: {type(e).__name__})"
        headers = {}

    finally:
        end_time = time.monotonic()
        with print_lock:
            print(f"--- Result {request_num} ({model_id} / ID: ...{unique_id[-6:]}) ---")
            print(f"  Status: {status}")
            if error_info:
                print(f"  Error: {error_info}")
            if status == "FAILED (429 Rate Limit)":
                print(f"  Retry After: {retry_after}")
            print(f"  Time Taken: {end_time - start_time:.3f}s")
            print(f"  Consumed Tokens (Reported): {consumed_tokens}") # Print consumed tokens
            print(f"  Headers:")
            print(f"    Remaining Tokens (TPM): {headers.get('x-ratelimit-remaining-tokens', 'N/A')}")
            print(f"    Reset Tokens (TPM):     {headers.get('x-ratelimit-reset-tokens', 'N/A')}")
            print(f"    Remaining Requests (RPD): {headers.get('x-ratelimit-remaining-requests', 'N/A')}")
            print(f"    Reset Requests (RPD):   {headers.get('x-ratelimit-reset-requests', 'N/A')}")
            print("-" * 35)
        # Return both headers and consumed tokens (if successful) for potential analysis
        return headers, consumed_tokens if status == "SUCCESS" else None

# --- Run the Test Again ---
print(f"Starting FINAL test: Sending {REQUESTS_TO_SEND} requests each to {MODEL_A} and {MODEL_B}.")
print("Using unique prompts and reporting ACTUAL consumed tokens from response 'usage' field.")
print("Observe Consumed Tokens vs Remaining Tokens (TPM).")
print("If Remaining TPM decreases by Consumed Tokens across models within a window -> Shared Pool.")
print("-" * 60)

threads = []
results_list = [] # Store tuples of (thread_name, headers, consumed_tokens)

# Define a wrapper to store results
def target_wrapper(model_id, req_num):
    headers, consumed = make_request(model_id, req_num)
    results_list.append((threading.current_thread().name, headers, consumed))

for i in range(REQUESTS_TO_SEND):
    t_a = threading.Thread(target=target_wrapper, args=(MODEL_A, i + 1), name=f"Thread-A-{i+1}")
    t_b = threading.Thread(target=target_wrapper, args=(MODEL_B, i + 1), name=f"Thread-B-{i+1}")
    threads.extend([t_a, t_b])

# Start threads
for t in threads:
    t.start()
    # Keep the slightly longer delay between starts
    # Allows observing if the API can handle near-simultaneous starts
    time.sleep(0.3)

# Wait for all threads
for t in threads:
    t.join()

print("-" * 60)
print("Final test finished.")

# Optional: Basic analysis of results could be added here later if needed
# e.g., sorting results_list by completion time and checking token decrement