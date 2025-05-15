# ParsingBloom

Version: 0.5.0

ParsingBloom is a proof-of-concept pipeline that turns raw into analytics-ready transaction records. The whole idea of ParsingBloom is to showcase the use of open-source transformer-based models (LLMs) in the context of building automated data pipelines that produce analytics-ready data. 


## Key ideas

| Principle | What it means in code |
|-----------|----------------------|
| **Deterministic output** | Every run over the same inputs yields bit-identical JSON; determinism tests (identity & structure) enforce this. |
| **Connector abstraction** | Gmail connector ships today; any IMAP, REST, or webhook source can be plugged in behind the same interface. |
| **Hybrid parsing** | LLM attempts strict JSON extraction. If it fails, deterministic regex fallback ensures fields are still produced. |
| **Dynamic schema support (0.5)** | Drop a YAML spec in `config/schemas/` and ParsingBloom will validate & export rows with exactly those fields. |


## Determinism tests

* **Identity tests** – byte-for-byte equality of two outputs.  
* **Structure tests** – identical field names & types even if values differ. 

If both these tests are passed, the pipeline is considered *self-consistent*.  

## Usage

### Local

<pre markdown>  bash:

chmod +x deploy/deploy_pb.sh            (to make sure its executable)

bash deploy/deploy_pb.sh
</pre>

**Options**

Note: These command options override whatever you have in config.yaml

- --gpus
    Enable CUDA/GPU flags (if your container or host has Nvidia drivers).

-   --force
    Reinstall dependencies and overwrite any existing build artifacts.

-   --gpus             Use GPU (sets PARSINGBLOOM_DEVICE=cuda)

-   --runs N           Number of replicates (default: 1)

-   --out-dir DIR      Base output directory (default: data/ if --runs=1; data/pipeline_executes/ if >1)

-   --schedule MODE    Daemon mode (only valid when --runs=1): hourly, daily, or cron per config

-   -h, --help         Show this help and exit

    Requires: Python 3.10 toolchain, Hugging Face token.

### Docker

export HF_API_TOKEN=hf_xxx

<pre markdown>  bash:

deploy/deploy_docker.sh
</pre>

**Options**

Same as local deployment options

### Determinism Testing Framework.

<pre markdown> bash:

chmod +x deploy/deploy_determinism.sh (to set permisions for shell script to be executable)
bash deploy/deploy_determinism.sh
</pre>

- --runs <n> where 
    Choosing the number of runs to use for statistics. n is an integer Default value is 30.

- --dir (optional)
    Choose directory plots will be added to. Default directory is 'data/determinism_tests'


Just run the plots

<pre markdown> 
python pipeline/determinism_plot.py --dir data/determinism_tests
</pre>

## Full explanation of system behavior


1. Authenticates to Gmail and fetches new messages matching your filter.  
2. Sends each email body to an LLM (OpenAI, llama-cpp or HuggingFace) with a strict JSON extraction prompt.  
3. If the LLM output is empty or fails a JSON-Schema validation, a fallback regex pass kicks in to extract date, amount, balance, account last-4 and description.  
4. Has dynamic schema creation capabilities. You create a a schema (in a .yaml file) and the system parses datas with those fields.  
5. A run is an instance of execution of the information scraping procedure. Writes a per-run CSV (`data/runs/{run_id}.csv`) and appends every row (including flagged ones) to a master CSV (`data/transactions.csv`). 
6. Tracks run metadata (start/end times, counts) and flags unparsed or ambiguous items for manual review.  
7. Offers a standalone monitoring script that computes z-scores on run sizes to detect anomalies.
8. System can be set to run like a daemon that remains alive until some termination condition is met (including manual termination of the process)........ Not yet thouse, will add this functionality.

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
    git clone https://github.com/you/ParsingBloom.git
    cd ParsingBloom
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
## Determinism Tests




## Sample Output Plots



## Possible next steps

    - Integrate a schema registry (Confluent or Git-backed JSON-Schema) and enforce versioning at ingest.

    - Enhance drift-detection with Great Expectations or dbt tests for schema and distribution checks.

    - Build a web UI to edit schedule, filters and account maps and push updates to the config store.

    - Implement staging + upsert loading into Postgres (or Snowflake if feasible), with automatic merging on email_id + timestamp.

    - Migrate secrets from keyring to Vault or a cloud KMS for production-grade security.

    - Extend monitoring to include field-level z-scores, null-rate alerts and regression testing against a golden email set.

    - Incorporate automated self-diagnostic capabilities for adaptive tuning using meta-analytics.
