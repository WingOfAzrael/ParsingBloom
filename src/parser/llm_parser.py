# Hybrid parser: LLM → validate → regex fallback.
# Supports OpenAI, llama-cpp, and HuggingFace 4 and 8-bit (and none) quantized models,
# and enforces/pivots around your known internal accounts from config.yaml.

import json
import logging
import regex as re
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from json import JSONDecoder, JSONDecodeError
from typing import Dict, Optional, Union, List

import keyring
from email.utils import parsedate_to_datetime
from jsonschema import validate
from transformers import pipeline, BitsAndBytesConfig
from llama_cpp import Llama

from utils.validators import validate_record, get_model
from utils.regex_fallback import dispatch as regex_dispatch
from parser.email_utils import extract_body
from core.classifier import TransactionClassifier
from core.models import Transaction
from pydantic import ValidationError, BaseModel
from core.config_loader import load_config


#Force debugging script to print to console
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,                         # print to console
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)
logger = logging.getLogger(__name__)
# ── Config & maps ──────────────────────────────────────────────────────────
CFG        = load_config()
PARSER_CFG = CFG.parser
PROVIDER   = PARSER_CFG.provider.lower()

# build internal-account maps
ACCOUNT_NAME_MAP = {}
ACCOUNT_NUM_MAP  = {}
ACCOUNT_CURRENCY_MAP = {}
duplicates = defaultdict(list)
for acct in CFG.accounts:
    raw    = acct.internal_account_number.strip()
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 4:
        continue
    last4 = digits[-4:]
    if last4 in ACCOUNT_NUM_MAP and ACCOUNT_NUM_MAP[last4] != raw:
        duplicates[last4].append(raw)
    ACCOUNT_NUM_MAP[last4]      = raw
    ACCOUNT_NAME_MAP[last4]     = acct.internal_entity.strip()
    ACCOUNT_CURRENCY_MAP[last4] = acct.currency.strip()
if duplicates:
    logging.warning("Duplicate last‐4 in config.accounts: %s", dict(duplicates))
assert ACCOUNT_NAME_MAP, "No accounts loaded—check `config.yaml` under `accounts:`"

# map merchant→last4 for swap
NAME_TO_LAST4 = {name.lower(): l4 for l4, name in ACCOUNT_NAME_MAP.items()}

# human‐readable account reference for prompts
REF_LINES = [
    f"- {ACCOUNT_NAME_MAP[l4]} ({ACCOUNT_NUM_MAP[l4]}) (last4={l4})"
    for l4 in ACCOUNT_NUM_MAP
]
ACCT_REF = "Your internal accounts:\n" + "\n".join(REF_LINES)

_JSON_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.S)   #regex library supports (?R) on python 3.10

def _extract_first_json(txt: str):
    m = _JSON_RE.search(txt)
    if not m:
        return None
    try:
        import json
        return json.loads(m.group(0))
    except Exception:
        return None

# Prompt template to confine LLM output to our schema. This is the schema chosen for your desired parsing data.
# ------------------------------------------------------------------ #
#  Build JSON schema block dynamically from YAML / Pydantic v2       #
# ------------------------------------------------------------------ #
ACTIVE_SCHEMA_NAME = CFG.active_schema               # e.g. "transactions"
RecordModel: type[BaseModel] = get_model(ACTIVE_SCHEMA_NAME)
TXN_SCHEMA  = RecordModel.model_json_schema()        # dict
schema_block = json.dumps(TXN_SCHEMA, indent=2)\
                    .replace("{", "{{")\
                    .replace("}", "}}")

#schema_block = json.dumps(TXN_SCHEMA, indent=2).replace("{", "{{").replace("}", "}}")
PROMPT_TMPL = f"""
{ACCT_REF}

You are a strict JSON generator.

Output **one** JSON object that satisfies the exact schema below.
No Markdown, no extra keys, no explanation—just the JSON.

Return ONLY JSON matching the schema, no markdown or other text.

<JSON_SCHEMA>
{schema_block}
</JSON_SCHEMA>

From: {{from_addr}}
Subject: {{subject}}

Body:
\"\"\"{{body}}\"\"\"

JSON:
"""

#PROMPT_TMPL = f"""
#{ACCT_REF}
#
#You are a strict JSON generator.
#
#Output **one** JSON object that satisfies the exact schema below.
#No Markdown, no extra keys, no explanation—just the JSON.
#
#Return ONLY JSON matching the schema, no markdown or other text.
#
#<JSON_SCHEMA>
#{schema_block}
#</JSON_SCHEMA>
#
#From: {{from_addr}}
#Subject: {{subject}}
#
#Body:
#\"\"\"{{body}}\"\"\"
#
#JSON:
#"""

# Balance regex extractor 
BALANCE_RE = re.compile(
    r"""Available\ balance [^\d]{0,20}? (?:R|ZAR)?\s* ([\d\.,]+)""",
    re.I | re.X,
)

def extract_available_balance(text: str) -> Optional[float]:
    m = BALANCE_RE.search(text)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except ValueError:
        return None

# ── Provider-specific initializers ────────────────────────────────────────
def _init_openai():
    import openai
    openai.api_key = keyring.get_password("email_agent", "openai_key")
    model_name = PARSER_CFG.openai_model or PARSER_CFG.model_id
    def _chat(prompt: str) -> str:
        resp = openai.ChatCompletion.create(
            model       = model_name,
            messages    = [{"role":"user","content":prompt}],
            temperature = PARSER_CFG.temperature,
            max_tokens  = PARSER_CFG.max_tokens,
        )
        return resp.choices[0].message.content
    return _chat

def _init_llamacpp():
    llama = Llama(
        model_path  = PARSER_CFG.model_path,
        n_ctx       = PARSER_CFG.n_ctx,
        temperature = PARSER_CFG.temperature,
        seed        = PARSER_CFG.seed,
    )
    return lambda prompt: llama(prompt).choices[0].text

# ── HuggingFace initializer (now parameterized) ─────────────────────────
def _init_hf(
    device: Union[str, int],
    quant_bits: int | None
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    # 1) Determine GPU vs CPU
    if isinstance(device, int):
        use_gpu = device >= 0
        device_str = f"cuda:{device}" if use_gpu else "cpu"
    else:
        device_str = str(device)
        use_gpu = device_str.lower().startswith("cuda")

    # 2) Resolve quant_bits
    qb = quant_bits if quant_bits is not None else PARSER_CFG.quant_bits

    # 3) Build quant config if needed
    quant_cfg = None
    if use_gpu:
        if qb == 4:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit           = True,
                bnb_4bit_quant_type    = PARSER_CFG.bnb_4bit_quant_type,
                bnb_double_quant       = PARSER_CFG.bnb_double_quant,
                bnb_4bit_compute_dtype = PARSER_CFG.bnb_4bit_compute_dtype,
            )
        elif qb == 8:
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit                     = True,
                llm_int8_enable_fp32_cpu_offload = PARSER_CFG.llm_int8_enable_fp32_cpu_offload,
            )
        elif qb is None:
            quant_cfg = None
        else:
            raise ValueError(f"Unsupported quant_bits={qb!r}, must be 4, 8, or None")

    # 4) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(PARSER_CFG.hf_model_id)
    if use_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            PARSER_CFG.hf_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            PARSER_CFG.hf_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    # 5) Ensure pad_token_id
    pad_id = (
        tokenizer.pad_token_id
        or getattr(tokenizer, "eos_token_id", None)
        or getattr(model.config, "eos_token_id", None)
    )
    if pad_id is None:
        raise ValueError("Cannot find pad_token_id or eos_token_id on tokenizer/model")
    tokenizer.pad_token_id    = pad_id
    tokenizer.padding_side    = getattr(tokenizer, "padding_side", "left")
    model.config.pad_token_id = pad_id

    # 6) Build HF pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=PARSER_CFG.hf_batch_size,
    )
    hf_pipeline.tokenizer.pad_token_id = hf_pipeline.tokenizer.pad_token_id or model.config.eos_token_id
    hf_pipeline.tokenizer.padding_side = "left"

    # 7) Return chat function
    def _chat(prompts: Union[str, List[str]], **generation_kwargs) -> Union[str, List[str]]:
        defaults = {
            "max_new_tokens":   PARSER_CFG.max_new_tokens,
            "do_sample":        False,
            "temperature":      PARSER_CFG.temperature,
            "return_full_text": True,
        }
        gen_kwargs = {**defaults, **generation_kwargs}
        results = hf_pipeline(prompts, **gen_kwargs)
        if isinstance(prompts, str):
            return results[0]["generated_text"]
        else:
            return [cand_list[0]["generated_text"] for cand_list in results]
    return _chat


class LLMParser:
    def __init__(
        self,
        device: Union[str, int, None]     = None,
        quant_bits: int | None             = None,
        provider: str | None               = None,
        **kwargs
    ):
        # 1) Provider
        self.provider = (provider or PROVIDER).lower()

        # 2) Resolve device & quant_bits
        env_dev = os.getenv("PARSINGFORGE_DEVICE") or os.getenv("PARSINGBLOOM_DEVICE")
        self.device = (
            device
            if device is not None
            else (env_dev or PARSER_CFG.device)
        )
        self.quant_bits = (
            quant_bits
            if quant_bits is not None
            else PARSER_CFG.quant_bits
        )

        # 3) Build the chat function
        if self.provider in ("huggingface", "hf"):
            self._chat = _init_hf(self.device, self.quant_bits)
        elif self.provider in ("openai", "oa"):
            self._chat = _init_openai()
        elif self.provider in ("llama-cpp", "llama_cpp"):
            self._chat = _init_llamacpp()
        else:
            raise ValueError(f"Unknown provider {self.provider!r}")

    def parse_batch(
        self,
        msgs: List[Dict],
        pdf_texts: List[Optional[str]],
        run_id: str = "",
    ) -> List[BaseModel]:
        logger = logging.getLogger(__name__)
        dec = JSONDecoder()

        # 1) Build records & prompts
        records = []
        for msg, pdf in zip(msgs, pdf_texts):
            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            # Canonical message ID: prefer RFC-822 header, else Gmail's ID
            msg_id_header = headers.get("Message-ID") or headers.get("Message-Id")
            canonical_id  = (msg_id_header.strip("<>") if msg_id_header else msg["id"]
            )
            body_text = extract_body(msg)
            if pdf:
                body_text += "\n\n" + pdf
            prompt = PROMPT_TMPL.format(
                from_addr=headers.get("From", ""),
                subject=headers.get("Subject", "")[:120],
                body=body_text
            )
            records.append({
                "msg": msg,
                "prompt": prompt,
                "canonical_id": canonical_id,   # NEW – RFC-822 or Gmail ID
                "gmail_id":   msg["id"],      # NEW – Gmail record ID
            })

        prompts = [r["prompt"] for r in records]
        all_outputs = self._chat(
            prompts,
            max_new_tokens   = PARSER_CFG.max_new_tokens,
            do_sample        = False,
            temperature      = PARSER_CFG.temperature,
            return_full_text = False,
            batch_size       = PARSER_CFG.hf_batch_size,
        )
        logger.debug("LLM raw outputs: %s", all_outputs[0:1])
        # 3) Decode or fallback
        txns: List[Optional[BaseModel]] = [None] * len(records)
        fallback_idxs: List[int] = []
        marker = "</JSON_SCHEMA>"
        for idx, (rec, out) in enumerate(zip(records, all_outputs)):
            raw = out if isinstance(out, str) else out.get("generated_text", "")

            data = _extract_first_json(raw)
            if data is None:
                logger.warning("No JSON found → regex fallback")
                txns.append(regex_dispatch(rec["msg"]))
                continue

            try:
                # inject canonical message ID
                #data["message_id"] = canonical_id
                data.setdefault("message_id", rec["canonical_id"])  # NEW
                data["message_id"] = rec["gmail_id"]
                data = validate_record(data, CFG.active_schema)
                txns[idx] = RecordModel(**{**data, "run_id": run_id})
            except ValidationError as e:
                # adapt regex_dispatch for arbitrary schemas.
                # For now, return raw JSON in RecordModel shape if possible.
                logger.debug("Schema validation failed → regex fallback: %s", e)
                fb = regex_dispatch(rec["msg"])
                setattr(fb, "run_id", run_id)           # ensure run_id present
                setattr(fb, "message_id", canonical_id)
                setattr(fb, "message_id", rec["gmail_id"]) 
                txns[idx] = fb

        fallback_idxs.clear()
        #Redundant fallback for any records that failed to parse. This helps if the LLM output is not valid JSON, but it is not desired behaviour because it turns ParsingBloom into a regex parser, and thats not the point.
        #if fallback_idxs:
        #    fb_prompts = [records[i]["prompt"] for i in fallback_idxs]
        #    fb_outputs = self._chat(
        #        fb_prompts,
        #        max_new_tokens   = PARSER_CFG.max_new_tokens,
        #        do_sample        = False,
        #        temperature      = PARSER_CFG.temperature,
        #        return_full_text = False,
        #        batch_size       = PARSER_CFG.hf_batch_size,
        #    )
        #    for idx, fb_out in zip(fallback_idxs, fb_outputs):
        #        fb_raw = fb_out if isinstance(fb_out, str) else fb_out.get("generated_text", "")
        #        brace  = fb_raw.find("{")
        #        if brace == -1:
        #            txns[idx] = regex_dispatch(records[idx]["msg"])
        #            continue
        #        
        #        data, _ = dec.raw_decode(fb_raw[brace:])
        #
        #        # 1) Try to coerce & validate against the active schema
        #        try:
        #            clean = validate_record(data, ACTIVE_SCHEMA_NAME)       # ← dynamic
        #            txns[idx] = RecordModel(**{**clean, "run_id": run_id})  # ← dynamic
        #            continue
        #        except ValidationError as e:
        #            logger.debug("Fallback JSON failed schema: %s", e)
        #
        #        # 2) Last resort: regex parser
        #        txns[idx] = regex_dispatch(records[idx]["msg"])

        return txns

    @staticmethod
    def _extract_sent_date(msg: Dict) -> str:
        for h in msg.get("payload", {}).get("headers", []):
            if h.get("name", "").lower() == "date":
                try:
                    dt = parsedate_to_datetime(h["value"])
                    return dt.astimezone().strftime("%Y-%m-%d")
                except:
                    pass
        if "internalDate" in msg:
            ts = int(msg["internalDate"]) / 1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
            return dt.strftime("%Y-%m-%d")
        raise ValueError("No Date header or internalDate")
