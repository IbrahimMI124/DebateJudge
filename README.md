# DebateJudge: Automated Debate Adjudication Pipeline

This project is a multi-module pipeline designed to automatically analyze a debate transcript, extract key claims, verify them against a knowledge corpus, and declare a winner based on factual accuracy, relevance, and consistency.

## Project Structure

The pipeline is orchestrated by `main.py` and consists of four main modules:

*   **Module 1 (Preprocessing):** Segments the raw debate transcript into individual argumentative statements.
*   **Module 2 (Claim Extraction):** Identifies and extracts structured, verifiable claims from each statement using a Hugging Face model.
*   **Module 3 (Verification):** Uses a Retrieval-Augmented Generation (RAG) approach to fact-check statistical claims. It retrieves relevant passages from a corpus and uses a local LLM (via Ollama) to judge the claim's accuracy.
*   **Module 4 (Judgement):** Scores each speaker on multiple metrics and determines the overall winner of the debate.

## Setup and Installation

Follow these steps precisely to set up the environment and run the pipeline.

### Step 1: Install System Dependencies (Ollama)

This project uses Ollama to run a local Large Language Model for fact-checking in Module 3.

1.  **Install the Ollama Application:** Open your terminal and run the official installation script.
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

2.  **Download the Required LLM:** The verifier module specifically requires the `phi3` model. Pull it using the following command. This is a large download (approx. 3.8 GB).
    ```bash
    ollama pull phi3
    ```
    The Ollama application will automatically start and run in the background after installation.

### Step 2: Install Python Libraries

This project requires several Python packages. Install them all using pip:

```bash
pip install torch transformers spacy pandas sentence-transformers faiss-cpu beautifulsoup4 lxml ollama huggingface_hub
```
*Note: `faiss-cpu` is recommended for general use. If you have a compatible NVIDIA GPU and CUDA installed, you can use `faiss-gpu` for better performance.*

### Step 3: Download NLP Model Data

After installing the Python libraries, download the necessary model for the `spacy` library used in Module 1.

```bash
python -m spacy download en_core_web_sm
```

### Step 4: Generate the Knowledge Corpus

Module 3 relies on a JSON file (`corpus.json`) created by scraping Wikipedia. If this file does not exist in the `module3/data/` directory, you must generate it by running the scraper script once.

```bash
python module3/scrape_corpus.py
```
*You only need to do this once. If `module3/data/corpus.json` already exists, you can skip this step.*

## How to Run the Pipeline

Once all setup steps are complete, you can run the entire debate judge pipeline with a single command from the project's root directory.

1.  **Ensure Input File Exists:** Make sure your debate transcript file is present. The script is configured to use `data/debate1.txt` by default.

2.  **Execute the Main Script:**
    ```bash
    python main.py
    ```

### What to Expect on First Run

The very first time you run the script, it will download and cache several pre-trained models from Hugging Face Hub for Modules 2 and 4. This may take a few minutes and requires an internet connection. **This is a one-time process.** Subsequent runs will be much faster as they will use the cached local files.

### Output

The pipeline will produce the following outputs in the project's root directory:

*   `output_module1.json`: A list of argumentative statements from the debate.
*   `output_module2.json`: A list of structured claims extracted from the statements.
*   `output_module3.json`: A list of verification results for each claim.
*   `output_final_judgement.json`: The final result, including speaker scores, consistency analysis, and the declared winner.

The final result will also be printed to the terminal upon completion.