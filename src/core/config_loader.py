# ────────────────────────────────────────────────────────────────────────────
# config/config_loader.py   (2025-05-10 patched – schema name fix)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, List

import yaml
from pydantic import BaseModel, Field, model_validator, field_validator


# Compute project root by walking up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow override via env var, fallback to <project_root>/config/config.yaml
DEFAULT_CONFIG_PATH = Path(
    os.getenv("PARSINGBLOOM_CONFIG", PROJECT_ROOT / "config" / "config.yaml")
)

def _expand(v: Any) -> Any:
    return Path(os.path.expanduser(v)).resolve() if isinstance(v, str) else v


# ── connector ───────────────────────────────────────────────────────────────
class ConnectorConfig(BaseModel):
    credentials_path: Path
    token_path:       Path
    max_results: Optional[int] = None
    query:      Optional[Union[str, Dict[str, Any]]] = None

    _norm = field_validator("credentials_path", "token_path", mode="before")(_expand)

    @field_validator("query", mode="before")
    def _canon_query(cls, v):
        if isinstance(v, dict):
            return " ".join(f"{k}:{v[k]}" for k in sorted(v))
        return v


# ── account ─────────────────────────────────────────────────────────────────
class AccountConfig(BaseModel):
    internal_account_number: str
    internal_entity:         str
    institution:             str
    currency:                str

# ── account_mask configuration ───────────────────────────────────────────────
class AccountMaskConfig(BaseModel):
    regex: str
    suffix_length: int

# ── paths (subscriptable) ───────────────────────────────────────────────────
class PathsConfig(BaseModel):
    data_dir:    Path
    invoice_dir: Path
    runs_csv:         Optional[Path] = None
    master_file:      Optional[Path] = None
    metadata_csv:     Optional[Path] = None
    flagged_csv:      Optional[Path] = None
    accounts_csv:     Optional[Path] = None

    _norm = field_validator('*', mode="before")(_expand)

    @model_validator(mode="before")
    def _alias(cls, v):
        m = {"runs_file": "runs_csv", "master_file": "master_csv"}
        for src, dst in m.items():
            if src in v and dst not in v:
                v[dst] = v[src]
        return v

    @model_validator(mode="after")
    def _defaults(cls, v):
        #d = v["data_dir"]
        d = v.data_dir
        defs = {
            "runs_csv":         d / "runs.csv",
            "master_file": d / "master.csv",
            "metadata_csv":     d / "metadata.csv",
            "flagged_csv":      d / "flagged_messages.csv",
            "accounts_csv":     Path(__file__).parent / "accounts.csv",
        }
        #for k, p in defs.items():
        #    v.setdefault(k, p)
        #return v
        for k, p in defs.items():
            if getattr(v, k) is None:
                setattr(v, k, p)
        return v

    def __getitem__(self, k): return getattr(self, k)
    def get(self, k, d=None): return getattr(self, k, d)

#Email filter config
class EmailFilterConfig(BaseModel):
    use_label:         bool   = False
    label_name:        str
    use_domain_filter: bool   = False
    sender_csv:        Path


# ── scraper / parser ────────────────────────────────────────────────────────
class ScraperConfig(BaseModel):
    initial_start_date: str
    overlap_hours: int = 0
    parse_pdfs:   bool = False
    save_attachments: bool = False
    pdf_password: Optional[str] = None


class ParserConfig(BaseModel):
    # Core provider selection
    provider:   Literal["openai", "llama-cpp", "huggingface"]

    # Model identifiers
    model_id:   str  = Field(..., description="HF or OpenAI model ID")
    hf_model_id: str = Field(..., description="HuggingFace model ID (same as model_id for HF)")

    # Local model paths
    model_path: str  = Field(..., description="Local Llama model binary path")

    # Device configuration
    device_map: str  = Field("auto", description='"auto" or "none"')
    device:     int  = Field(0,      description="GPU index; 0 = first GPU, -1 = CPU")

    #Batch loading size
    hf_batch_size: int  = Field(1, description="Batch size for local models")

    # Generation parameters
    max_tokens:       int    = Field(512,  description="OpenAI max_tokens or HF max_length")
    max_new_tokens:   int    = Field(512,  description="HF max_new_tokens")
    n_ctx:            int    = Field(2048, description="Context window size for local models")
    temperature:      float  = Field(0.0,  description="Sampling temperature")
    seed:             int    = Field(42,   description="RNG seed for reproducibility")

    # OpenAI-specific
    openai_model:     str    = Field("gpt-4", description="OpenAI model name")

    # Quantization settings for bitsandbytes
    quant_bits:            int    = Field(4,       description="Number of quantization bits")
    bnb_4bit_quant_type:   str    = Field("nf4",   description="BitsAndBytes 4-bit quant type")
    bnb_double_quant:      bool   = Field(True,     description="Enable BitsAndBytes double quant")
    bnb_4bit_compute_dtype:str   = Field("float16",description="Compute dtype for 4-bit matmuls")

    def __getitem__(self, k):
        return getattr(self, k)

    def get(self, k, d=None):
        return getattr(self, k, d)


# ── database (subscriptable) ────────────────────────────────────────────────
class DatabaseConfig(BaseModel):
    type:      Optional[str] = None
    host:      Optional[str] = None
    port:      Optional[int] = None
    dbname:    Optional[str] = None
    user:      Optional[str] = None
    table:     Optional[str] = "transactions"
    db_schema: Optional[str] = Field("public", alias="schema")

    class Config:
        allow_population_by_field_name = True

    def __getitem__(self, k):
        if k == "schema":
            return self.db_schema
        return getattr(self, k)

    def get(self, k, d=None):
        if k == "schema":
            return getattr(self, "db_schema", d)
        return getattr(self, k, d)

# ── determinism config ───────────────────────────────────────────────────────
class DeterminismConfig(BaseModel):
    struct_threshold: float = Field(..., description="Minimum structural determinism rate to pass")
    id_threshold:    float = Field(..., description="Minimum lower-bound CI for identity determinism to pass")
    alpha:           float = Field(..., description="Alpha level for Clopper–Pearson intervals")
    latency_cv_threshold: float = Field(..., description="Max allowed coefficient of variation for latency")


# ── full config ─────────────────────────────────────────────────────────────
class FullConfig(BaseModel):
    default_connector: Optional[str] = None
    active_schema:    Optional[str] = None
    connectors:        Dict[str, ConnectorConfig]
    account_mask:      AccountMaskConfig
    accounts:          List[AccountConfig]
    determinism:      DeterminismConfig
    paths:             PathsConfig
    scraper:           ScraperConfig
    parser:            ParserConfig
    database:          DatabaseConfig = DatabaseConfig()
    email_filter:       EmailFilterConfig             
    scheduler:         Dict[str, Any] = {}
    pdf_passwords:     Dict[str, str] = {}
    synonyms:          Dict[str, List[str]] = {}

    def __getitem__(self, k): return getattr(self, k)
    def get(self, k, d=None): return getattr(self, k, d)

    @field_validator("synonyms", mode="before")
    def _norm_syn(cls, v):
        out = {}
        for f, val in (v or {}).items():
            if val is None:
                out[f] = []
            elif isinstance(val, list):
                out[f] = [str(x) for x in val]
            else:
                out[f] = [str(val)]
        return out


# ── loader singleton ────────────────────────────────────────────────────────
_CFG: Optional[FullConfig] = None

def load_config(path: Union[str, Path] | None = None) -> FullConfig:
    """
    Load the main ParsingBloom config, with these fallbacks:
      1) If `path` is provided, use it.
      2) Else if env var PARSINGBLOOM_CONFIG (or PARSINGFORGE_CONFIG) is set, use that.
      3) Else look under <project_root>/config/config.yaml.
    Caches the result in _CFG so subsequent calls are cheap.
    Applies a default_connector if only one is defined,
    and respects PARSINGBLOOM_DEVICE / PARSINGFORGE_DEVICE overrides.
    """
    global _CFG
    if _CFG is not None:
        return _CFG

    # 1. Determine project root (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]

    # 2. Build default path: env var override or config/config.yaml in project root
    env_path = os.getenv("PARSINGBLOOM_CONFIG") or os.getenv("PARSINGFORGE_CONFIG")
    default_cfg_path = Path(env_path) if env_path else project_root / "config" / "config.yaml"

    # 3. Final path: explicit argument wins, then default
    cfg_path = Path(path) if path else default_cfg_path
    cfg_path = _expand(cfg_path)  # your existing helper for ~, env vars, etc.

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path!s}")

    # 4. Load YAML
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 5. Default connector inference
    if "default_connector" not in raw:
        conns = raw.get("connectors", {})
        if len(conns) == 1:
            raw["default_connector"] = next(iter(conns))
        else:
            raise ValueError(
                "config.yaml: 'default_connector' missing and multiple connectors found."
            )

    # 6. Build the typed config
    cfg = FullConfig(**raw)

    # 7. Device override (CPU vs CUDA)
    dev = os.getenv("PARSINGBLOOM_DEVICE") or os.getenv("PARSINGFORGE_DEVICE")
    if dev:
        dev_lower = dev.lower()
        if dev_lower == "cpu":
            cfg.parser.device, cfg.parser.device_map = -1, "none"
        elif dev_lower.startswith("cuda"):
            idx = int(dev.split(":", 1)[1]) if ":" in dev else 0
            cfg.parser.device, cfg.parser.device_map = idx, "auto"

    # 8. Cache and return
    _CFG = cfg
    return cfg
