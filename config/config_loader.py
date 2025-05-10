# ────────────────────────────────────────────────────────────────────────────
# config/config_loader.py   (2025-05-10 patched – schema name fix)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, List

import yaml
from pydantic import BaseModel, Field, validator, root_validator


def _expand(v: Any) -> Any:
    return Path(os.path.expanduser(v)).resolve() if isinstance(v, str) else v


# ── connector ───────────────────────────────────────────────────────────────
class ConnectorConfig(BaseModel):
    credentials_path: Path
    token_path:       Path
    max_results: Optional[int] = None
    query:      Optional[Union[str, Dict[str, Any]]] = None

    _norm   = validator("credentials_path", "token_path",
                        pre=True, allow_reuse=True)(_expand)

    @validator("query", pre=True, allow_reuse=True)
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


# ── paths (subscriptable) ───────────────────────────────────────────────────
class PathsConfig(BaseModel):
    data_dir:    Path
    invoice_dir: Path
    runs_csv:         Optional[Path] = None
    transactions_csv: Optional[Path] = None
    metadata_csv:     Optional[Path] = None
    flagged_csv:      Optional[Path] = None
    accounts_csv:     Optional[Path] = None

    _norm = validator('*', pre=True, allow_reuse=True)(_expand)

    @root_validator(pre=True)
    def _alias(cls, v):
        m = {"runs_file": "runs_csv", "transactions_file": "transactions_csv"}
        for src, dst in m.items():
            if src in v and dst not in v:
                v[dst] = v[src]
        return v

    @root_validator
    def _defaults(cls, v):
        d = v["data_dir"]
        defs = {
            "runs_csv":         d / "runs.csv",
            "transactions_csv": d / "transactions.csv",
            "metadata_csv":     d / "metadata.csv",
            "flagged_csv":      d / "flagged_messages.csv",
            "accounts_csv":     Path(__file__).parent / "accounts.csv",
        }
        for k, p in defs.items():
            v.setdefault(k, p)
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
    save_attachments: bool = True
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


# ── full config ─────────────────────────────────────────────────────────────
class FullConfig(BaseModel):
    default_connector: Optional[str] = None
    connectors:        Dict[str, ConnectorConfig]
    accounts:          List[AccountConfig]
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

    @validator("synonyms", pre=True, allow_reuse=True)
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
    global _CFG
    if _CFG is not None:
        return _CFG

    path = _expand(path or Path(__file__).parent / "config.yaml")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "default_connector" not in raw:
        conns = raw.get("connectors", {})
        if len(conns) == 1:
            raw["default_connector"] = next(iter(conns))
        else:
            raise ValueError("config.yaml: 'default_connector' missing and multiple connectors found.")

    cfg = FullConfig(**raw)

    dev = os.getenv("PARSINGBLOOM_DEVICE") or os.getenv("PARSINGFORGE_DEVICE")
    if dev:
        if dev.lower() == "cpu":
            cfg.parser.device, cfg.parser.device_map = -1, "none"
        elif dev.lower().startswith("cuda"):
            idx = int(dev.split(":", 1)[1]) if ":" in dev else 0
            cfg.parser.device, cfg.parser.device_map = idx, "auto"

    _CFG = cfg
    return cfg
