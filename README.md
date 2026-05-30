# Firespeaker

An intelligent, context-aware audiobook creation platform leveraging state-of-the-art open-source Natural Language Processing (NLP) and zero-shot generative Speech Synthesis.

Firespeaker segments raw manuscripts, performs character entity resolution (via neural cross-context coreference `xCoRe`), extracts dialogue, maps emotional sentiments, and generates highly expressive character-consistent audio tracks using `XTTS-v2` and `Suno Bark`.

---

## 🛠️ Environment Setup & Installation

Follow these steps to set up your environment and install all necessary dependencies.

### 1. Initialize and Activate Virtual Environment

It is highly recommended to use a clean **Conda** or **venv** environment running Python 3.10 or 3.11.

#### Using Conda (Recommended):
```bash
# Create a dedicated Conda environment
conda create -n firespeaker python=3.10 -y

# Activate the environment
conda activate firespeaker
```

#### Using venv:
```bash
# Create a standard virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

---

### 2. Install Dependencies

Install all core, synthesis, testing, and cloud integration dependencies using the newly created `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> [!TIP]
> **GPU / CUDA Acceleration (Recommended):**
> If you have an NVIDIA GPU, make sure you install a CUDA-enabled version of PyTorch to enable fast inference:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 3. Download the NLP Language Model

The Firespeaker pipeline uses spaCy's large English model for typographic normalization and fallback character attributions. Download it after installing requirements:

```bash
python -m spacy download en_core_web_lg
```

---

## 🚀 Running the GUI Server

Once the dependencies are installed and the spaCy model is downloaded, boot up the local GUI server to interact with the web dashboard:

```bash
python src/gui_server.py
```

Open your browser and navigate to:
👉 **[http://localhost:8082](http://localhost:8082)**

---

## 📂 Project Architecture

*   `src/`: Primary codebase
    *   `gui_server.py`: Interactive 4-tab dashboard backend
    *   `nlp_analyzer.py`: Text pipeline, segmentation, VADER sentiment, & coreference fallbacks
    *   `spatial_memory.py`: Relational & ChromaDB vector "Memory Palace"
    *   `voice_synthesizer.py`: Deep learning speaker engine (XTTS-v2 / Bark Small)
    *   `audio_mixer.py`: Multi-track wav mixer and overlay engine
    *   `main.py`: Command-line pipeline execution entrypoint
*   `voice_synthesis_testing/`: Audio benchmarking, evaluations, and QA metrics
*   `nlp-testing/`: API integration and experimentation notebooks
