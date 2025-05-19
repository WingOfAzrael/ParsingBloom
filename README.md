
# ParsingBloom

Version: 0.5.0



## Table of Contents
- [Introduction](#introduction)
- [Full explanation of system behavior](#FullExplanationOfSystemBehavior)
- [Technologies Used ](#TechnologiesUsed )
- [Deployment ](#Deployment )
- [Determinism Tests](#DeterminismTests)
- [Deployment ](#Deployment )

## Introduction

ParsingBloom is a proof-of-concept pipeline that turns raw into analytics-ready transaction records. The whole idea of ParsingBloom is to showcase the use of open-source transformer-based models (LLMs) in the context of building automated data pipelines that produce analytics-ready data. 

## Key ideas

| Principle | What it means in code |
|-----------|----------------------|
| **Deterministic output** | Every run over the same inputs yields bit-identical JSON; determinism tests (identity & structure) enforce this. |
| **Connector abstraction** | Gmail connector ships today; any IMAP, REST, or webhook source can be plugged in behind the same interface. |
| **Hybrid parsing** | LLM attempts strict JSON extraction. If it fails, deterministic regex fallback ensures fields are still produced. |
| **Dynamic schema support (0.5)** | Drop a YAML spec in `config/schemas/` and ParsingBloom will validate & export rows with exactly those fields. |




## System behavior


1. Authenticates to Gmail and fetches new messages matching your filter.  
2. Sends each email body to an LLM (OpenAI, llama-cpp or HuggingFace) with a strict JSON extraction prompt.  
3. If the LLM output is empty or fails a JSON-Schema validation, a fallback regex pass kicks in to extract date, amount, balance, account last-4 and description.  
4. Has dynamic schema creation capabilities. You create a a schema (in a .yaml file) and the system parses datas with those fields.  
5. A run is an instance of execution of the information scraping procedure. Writes a per-run CSV (`data/runs/{run_id}.csv`) and appends every row (including flagged ones) to a master CSV (`data/transactions.csv`). 
6. Tracks run metadata (start/end times, counts) and flags unparsed or ambiguous items for manual review.  
7. Offers a standalone monitoring script that computes z-scores on run sizes to detect anomalies.
8. System can be set to run like a daemon that remains alive until some termination condition is met (including manual termination of the process)........ Not yet thouse, will add this functionality.



## Technologies Used

- **Huggingface transformers**: In particular, "meta-llama/Llama-3.2-3B-Instruct", for LLM parsing. Capabilities for OpenAI, llama-cpp-python APIs are built-in but untested. 
- **Schema validation**: Pydantic
- **The rest**: Pretty much standard python libraries for various tasks. 

## Usage

Note: System built and tested on a Unix-based OS, so use a Virtual Machine (VM) to deploy in the following way if on Windows.

1. **Clone the repo and install required modules**


   <pre markdown> 
    git clone https://github.com/you/ParsingBloom.git
    cd ParsingBloom
    </pre>

    Create a Python virtual environment & install poetry
    <pre markdown> 
    conda create venv
    conda activate venv
    pip install poetry
     </pre>

     Note: Poetry is used for dependency management, so rather just install this and let the project .toml file do the installation.

2. **Configure** 

    Note that default configuration (currently in code) assumes at least an NVIDIA RTX 2080 Super. We have quentised (reduced numerical precision of weights) the **Huggingface transformers** because while its not the biggest model, running locally unquantised (or with CPU) led to killed processes or hanging due to insufficient memory. 

    We note that fairly decent compute is required to perform this task using open-source models. If you use the OpenAI api, this is not a concern; rather, financial cost becomes a concern. This is the tradeoff.

    * You can play around with whatever configurations in config/config.yaml. 

    * Update Gmail paths (credentials_path, token_path), OpenAI keyring settings, schedule.cron and env_var.

    * Drop your config/credentials.json (Gmail OAuth2 secrets) in place.

    * Get access to whatevr model you'd like to use on Huggingface (most well integrated API at this stage) and use "export HF_API_TOKEN=hf_xxx". Or use whichever other model you like. 

### Local Deployment

<pre markdown>  

chmod +x deploy/deploy_pb.sh            (to make sure its executable)

deploy/deploy_pb.sh
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

### Docker Deployment



<pre markdown> 

deploy/deploy_docker.sh
</pre>

**Options**

Same as local deployment options

### Determinism Testing Deployment.

<pre markdown> 

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


## Determinism Tests 

### Depth-of-Testing Tiers

ParsingBloom ships with Tier 1 harness scripts (`src/analysis/determinism_test.py`).  
Upgrade paths:

Tiers exist that correspond to the quality of outputs based on their robustness. It is possible to perform diagnostic tests to prove this. The following table is helpful:

| Tier | Scope & Goal | Required Tests | When to Use | Effort ▼ |
|------|--------------|----------------|-------------|-----------|
| **1 Audit-Ready** | Prove outputs are stable & timely | • 30 × runs on 5–10 emails<br>• 100 % field-exact match<br>• Measure runtime mean ± σ; ensure CV < 5 %<br>• CI fails if modal-JSON changes | Most analytics engagements with standard SLAs | **Low** – straightforward CI + dashboards |
| **2 Engineering-Grade** | Catch subtle nondeterminism & environment drift | • SHA-256 hashing of prompts & raw LLM outputs<br>• Cross-env spot-checks (CPU vs GPU, 4/8-bit)<br>• Golden-file byte diff in CI | Larger teams, frequent infra/model churn | **Medium** – modest extra scripting + golden files |
| **3 Regulated-Grade** | Meet strict audit/compliance & SLA requirements | • Multi-GPU/CPU family grid tests<br>• Stochastic perturbation envelope (do_sample=True)<br>• Daily/weekly 3σ control charts, auto alerts | Finance, healthcare, government, or other high-assurance domains | **High** – infra orchestration + continuous monitoring |

### Underlying Theory

**Structural vs. identity determinism**

* **Identity tests** – byte-for-byte equality of two outputs (justifies Tier 1)
* **Structure tests** – identical field names & types even if values differ (justifies tier 2)

If both these tests are passed, the pipeline is considered *self-consistent*.  



### Corresponding


## Sample Output Plots

### Actual Analytics


### Diagnostic Analytics



## Possible next steps

    - Integrate a schema registry (Confluent or Git-backed JSON-Schema) and enforce versioning at ingest.

    - Enhance drift-detection with Great Expectations or dbt tests for schema and distribution checks.

    - Build a web UI to edit schedule, filters and account maps and push updates to the config store.

    - Implement staging + upsert loading into Postgres (or Snowflake if feasible), with automatic merging on email_id + timestamp.

    - Migrate secrets from keyring to Vault or a cloud KMS for production-grade security.

    - Extend monitoring to include field-level z-scores, null-rate alerts and regression testing against a golden email set.

    - Incorporate automated self-diagnostic capabilities for adaptive tuning using meta-analytics.


