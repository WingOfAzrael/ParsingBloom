# ParsingForge

Version: 0.2.1

ParsingForge is a proof-of-concept pipeline that turns banking emails into analytics-ready transaction records. The whole idea of ParsingForge is to showcase the use of open-source transformer-based models (LLMs) in the context of building automated data pipelines that produce analytics-ready data. The key consideration is that the system should provide deterministic parsing of either email bodies or invoices into datasets. 



## Quick start

```bash

chmod +x deploy/local_deploy.sh (to set permisions for shell script to be executable)
bash deploy/local_deploy.sh

    Requires: Python 3.10 toolchain, Hugging Face token.

Docker

export HF_API_TOKEN=hf_xxx
bash deploy/docker_deploy.sh

Runs an hourly scheduler inside the container.
```


## Full explanation of system behavior


1. Authenticates to Gmail and fetches new messages matching your filter.  
2. Sends each email body to an LLM (OpenAI, llama-cpp or HuggingFace) with a strict JSON extraction prompt.  
3. If the LLM output is empty or fails a JSON-Schema validation, a fallback regex pass kicks in to extract date, amount, balance, account last-4 and description.  
4. Builds a `Transaction` object, classifies it (personal/business/unclassified), and saves any PDF attachments via a pdfplumber stub.  
5. A run is an instance of execution of the information scraping procedure. Writes a per-run CSV (`data/runs/{run_id}.csv`) and appends every row (including flagged ones) to a master CSV (`data/transactions.csv`). 
6. Tracks run metadata (start/end times, counts) and flags unparsed or ambiguous items for manual review.  
7. Offers a standalone monitoring script that computes z-scores on run sizes to detect anomalies.
8. System runs like a daemon that remains alive until some termination condition is met (including manual termination of the process).

## Systems key features and problems they solve

- **Configuration-driven scheduling** with APScheduler and cron syntax, overrideable via an env var, so you can adjust scraping run frequency at runtime or via a UI later (no UI incorporated in this iteration).  
- **Hybrid parsing** (LLM-first + deterministic regex fallback) to maximise the probability that dimensions for datasets are correctly extracted.  
- **Per-run + master CSV exports** for full auditability, idempotent loading and easy backfilling.  
- **Attachment handling** stubbed via pdfplumber. More sophisticated methods an afterthought at this stage.  
- **Lightweight drift monitoring** decoupled from parsing logic, so you can catch data anomalies in COntinuous Integration (CI) or on a schedule without bloating the core pipeline.  
- **Flagging mechanism** for ambiguous last-4 matches or failed parses, ensuring no transaction is ever lost.
- **Self diagnosis** with analytics on system meta-data (I like to call this meta-analytics).

## Technologies used

- **Huggingface transformers**: In particular, "meta-llama/Llama-3.2-3B-Instruct", for LLM parsing. Capabilities for OpenAI, llama-cpp-python APIs are built-in but untested. 
- **The rest**: Pretty much standard python libraries for various tasks. 

## Manual deployment instructions (if quickstart is too convenient)

Note: System built and tested on a Unix-based OS, so use a Virtual Machine (VM) to deploy in the following way if on Windows.

1. **Clone the repo and install required modules**  
   <pre markdown> bash:
    git clone https://github.com/you/ParsingForge.git
    cd ParsingForge
    </pre>

    Create a Python virtual environment & install the dependencies
    <pre markdown> bash:
    conda create venv
    conda activate venv
    pip install -r requirements.txt
     </pre>

2. **Configure** 

    Note that default configuration (currently in code) assumes at least an NVIDIA RTX 2080 Super. We have quentised (reduced numerical precision of weights) the **Huggingface transformers** because while its not the biggest model, running locally unquantised led to killed process due to insufficient memory. 
    We note that fairly decent compute is required to perform this task using open-source models. If you use the OpenAI api, this is not a concern; rather, financial cost becomes a concern. This is the tradeoff.

    Copy config/config.yaml example → config/config.yaml. You can play around with whatever configurations. 

    Update Gmail paths (credentials_path, token_path), OpenAI keyring settings, schedule.cron and env_var.

    Drop your config/credentials.json (Gmail OAuth2 secrets) in place.

3. **Run the scheduler**
    <pre markdown>bash
    python schedule_runner.py
    </pre>

    Override on the fly:
    <pre markdown>bash
    SCHEDULE_CRON="0 0 * * *" python schedule_runner.py
    </pre>
    One-off/backfill runs
    <pre markdown> bash:
    python pipeline/run_pipeline.py --start-date 2023-01-01
     </pre>
4.  **Possible next steps**

    - Integrate a schema registry (Confluent or Git-backed JSON-Schema) and enforce versioning at ingest.

    - Swap in Google Document AI behind the PdfExtractor interface to pull invoice numbers, dates and totals.

    - Enhance drift-detection with Great Expectations or dbt tests for schema and distribution checks.

    - Build a web UI to edit schedule, filters and account maps and push updates to the config store.

    - Implement staging + upsert loading into Postgres (or Snowflake if feasible), with automatic merging on email_id + timestamp.

    - Migrate secrets from keyring to Vault or a cloud KMS for production-grade security.

    - Extend monitoring to include field-level z-scores, null-rate alerts and regression testing against a golden email set.

    - Incorporate automated self-diagnostic capabilities for adaptive tuning using meta-analytics.
