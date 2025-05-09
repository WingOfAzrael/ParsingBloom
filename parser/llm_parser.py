
"""
Hybrid parser: LLM → validate → regex fallback.
Supports OpenAI, llama-cpp, and HuggingFace 4-bit quantized models,
and enforces/pivots around your known internal accounts from config.yaml.
"""
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from json import JSONDecoder

import keyring
from email.utils import parsedate_to_datetime
from jsonschema import validate
from transformers import pipeline, BitsAndBytesConfig
from llama_cpp import Llama

from utils.validators import TXN_SCHEMA, validate_transaction_data
from utils.regex_fallback import dispatch as regex_dispatch
from parser.email_utils import extract_body
from core.classifier import TransactionClassifier
from core.models import Transaction
from config.config_loader import load_config

# ── Config & maps ──────────────────────────────────────────────────────────
CFG        = load_config("config/config.yaml")
PARSER_CFG = CFG["parser"]
PROVIDER   = PARSER_CFG["provider"].lower()

ACCOUNT_NAME_MAP, ACCOUNT_NUM_MAP, ACCOUNT_CURRENCY_MAP = {}, {}, {}
dup = defaultdict(list)
for acct in CFG.get("accounts", []):
    raw    = acct["internal_account_number"].strip()
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 4:
        continue
    last4 = digits[-4:]
    if last4 in ACCOUNT_NUM_MAP and ACCOUNT_NUM_MAP[last4] != raw:
        dup[last4].append(raw)
    ACCOUNT_NUM_MAP[last4]      = raw
    ACCOUNT_NAME_MAP[last4]     = acct["internal_entity"].strip()
    ACCOUNT_CURRENCY_MAP[last4] = acct["currency"].strip()
if dup:
    logging.warning("Duplicate last‐4 in config.accounts: %s", dict(dup))
assert ACCOUNT_NAME_MAP, "No accounts loaded—check `config.yaml` under `accounts:`"

NAME_TO_LAST4 = { name.lower(): l4 for l4, name in ACCOUNT_NAME_MAP.items() }

REF_LINES = [
    f"- {ACCOUNT_NAME_MAP[l4]} ({ACCOUNT_NUM_MAP[l4]}) (last4={l4})"
    for l4 in ACCOUNT_NUM_MAP
]
ACCT_REF = "Your internal accounts:\n" + "\n".join(REF_LINES)

# embed schema for the LLM
schema_block = json.dumps(TXN_SCHEMA, indent=2).replace("{", "{{").replace("}", "}}")
PROMPT_TMPL = f"""
{ACCT_REF}

Return ONLY JSON matching the schema.

<JSON_SCHEMA>
{schema_block}
</JSON_SCHEMA>

From: {{from_addr}}
Subject: {{subject}}

Body:
\"\"\"{{body}}\"\"\"

JSON:
"""

# ── Balance extractor ───────────────────────────────────────────────────────
BALANCE_RE = re.compile(
    r"""Available\ balance [^\d]{0,20}? (?:R|ZAR)?\s* ([\d\.,]+)""",
    re.I|re.X,
)
def extract_available_balance(text: str) -> float | None:
    m = BALANCE_RE.search(text)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except ValueError:
        return None

# ── LLM backends ───────────────────────────────────────────────────────────
def _init_openai():
    import openai
    client = openai.Client(
        api_key=keyring.get_password(
            CFG["openai"]["keyring_service"],
            CFG["openai"]["keyring_user"],
        ),
        base_url=CFG["openai"]["api_base"] or None,
    )
    model_name = PARSER_CFG.get("openai_model") or PARSER_CFG.get("model")
    def _chat(prompt: str) -> str:
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=PARSER_CFG.get("temperature", 0.0),
            top_p=1.0,
            n=1,
        ).choices[0].message.content
    return _chat

def _init_llamacpp():
    llama = Llama(
        model_path  = PARSER_CFG["model_path"],
        n_ctx       = PARSER_CFG.get("n_ctx", 2048),
        temperature = PARSER_CFG.get("temperature", 0.0),
        seed        = PARSER_CFG.get("seed", 42),
    )
    def _chat(prompt: str) -> str:
        return llama(prompt).choices[0].text
    return _chat

def _init_hf():
    quant_cfg = None
    if (qb := PARSER_CFG.get("quant_bits")):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit         = (qb == 4),
            bnb_4bit_quant_type  = PARSER_CFG.get("bnb_4bit_quant_type", "nf4"),
            bnb_double_quant     = PARSER_CFG.get("bnb_double_quant", False),
            bnb_4bit_compute_dtype = __import__("torch").float16,
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = PARSER_CFG["hf_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map          = PARSER_CFG.get("device_map", "auto"),
        quantization_config = quant_cfg,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    max_tokens = PARSER_CFG.get("max_new_tokens", 512)
    def _chat(prompt: str) -> str:
        out = pipe(
            prompt,
            max_new_tokens = max_tokens,
            do_sample      = False,
            temperature    = 0.0,
            top_k          = 1,
            top_p          = 1.0,
        )
        return out[0]["generated_text"]
    return _chat

CHAT = {"openai": _init_openai, "llamacpp": _init_llamacpp, "hf": _init_hf}[PROVIDER]()

# ── Parser ─────────────────────────────────────────────────────────────────
class LLMParser:
    def __init__(self):
        self.clf = TransactionClassifier()

    def parse(self, msg: dict) -> Transaction:
        headers   = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        from_addr = headers.get("From", "")
        subject   = headers.get("Subject", "")
        body      = extract_body(msg)

        prompt = PROMPT_TMPL.format(from_addr=from_addr, subject=subject[:120], body=body)
        data   = self._llm_pass(prompt, body)

        # force & pivot on internal account
        orig = data.get("account_number","")
        digits = re.sub(r"\D","",orig)
        if len(digits)>=4 and (l4:=digits[-4:]) in ACCOUNT_NUM_MAP:
            forced_acc = ACCOUNT_NUM_MAP[l4]
            forced_ent = ACCOUNT_NAME_MAP[l4]
            data["account_number"] = forced_acc
            data["account_name"]   = forced_ent
            data["currency"]       = ACCOUNT_CURRENCY_MAP.get(l4, data.get("currency",""))
            if forced_acc != orig:
                note = (
                    f"Use internal_account_number `{forced_acc}` "
                    f"and internal_entity `{forced_ent}` to re-extract all fields."
                )
                data = self._llm_pass(prompt + "\n\n" + note, body)

        # sent-date override
        try:
            data["date"] = self._extract_sent_date(msg)
        except ValueError:
            logging.warning("Date extraction failed")

        # description fallback
        if data.get("external_entity"):
            data["description"] = data["external_entity"]

        # balance fallback
        if data.get("available_balance") is None:
            bal = extract_available_balance(body) or extract_available_balance(subject)
            if bal is not None:
                data["available_balance"] = bal

        # external→internal swap if merchant matches
        orig_int, orig_ext = data.get("account_name",""), data.get("external_entity","")
        for name_lower, l4 in NAME_TO_LAST4.items():
            if name_lower in (orig_ext or "").lower() and orig_int.lower()!=name_lower:
                data["account_name"]    = ACCOUNT_NAME_MAP[l4]
                data["account_number"]  = ACCOUNT_NUM_MAP[l4]
                data["currency"]        = ACCOUNT_CURRENCY_MAP.get(l4, data.get("currency",""))
                data["external_entity"] = orig_int
                break

        # regex fallback if empty
        if not data:
            data = regex_dispatch(body)
            if not data:
                raise ValueError(f"Unable to extract fields from email {msg.get('id')}")

        # validate *after* date & all overrides
        validate(data, TXN_SCHEMA)
        validate_transaction_data(data)

        # build Transaction with `timestamp`, not `date`
        return Transaction(
            timestamp          = datetime.strptime(data["date"], "%Y-%m-%d"),
            account_name       = data["account_name"],
            account_number     = data["account_number"],
            institution        = data.get("institution",""),
            external_entity    = data.get("external_entity",""),
            amount             = float(data["amount"]),
            available_balance  = float(data["available_balance"]) if data.get("available_balance") else None,
            currency           = data.get("currency",""),
            description        = data.get("description",""),
            transaction_type   = "unclassified",
            source_email       = from_addr,
            email_id           = msg.get("id",""),
        )

    def parse_text(self, msg: dict, body: str) -> Transaction:
        # identical to parse(), using PDF‐extracted body
        data = self.parse(msg)  # or duplicate the same logic, substituting `body`
        return data

    def _llm_pass(self, prompt: str, body: str) -> dict:
        raw = CHAT(prompt)
        # skip prompt’s own schema block
        idx   = raw.find("</JSON_SCHEMA>")
        start = raw.find("{", idx + len("</JSON_SCHEMA>")) if idx!=-1 else raw.find("{")
        if start<0:
            raise ValueError("No JSON found in LLM response")
        data, _ = JSONDecoder().raw_decode(raw[start:])
        return data

    @staticmethod
    def _extract_sent_date(msg: dict) -> str:
        for h in msg.get("payload",{}).get("headers",[]):
            if h.get("name","").lower()=="date":
                try:
                    dt = parsedate_to_datetime(h["value"])
                    return dt.astimezone().strftime("%Y-%m-%d")
                except:
                    pass
        if "internalDate" in msg:
            ts = int(msg["internalDate"])/1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
            return dt.strftime("%Y-%m-%d")
        raise ValueError("No Date header or internalDate")