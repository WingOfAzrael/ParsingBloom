import os
import csv
import json
import base64
from datetime import datetime
from pathlib import Path
import yaml
import keyring

from core.models import Transaction
from parser.base import BankEmailParser

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected, using CPU.")

# Load config
cfg = yaml.safe_load(open("config/config.yaml"))

#here we load in account information to guide LLM parsing.
acct_path = cfg['paths']['accounts_csv']
internal_accounts = []
with open(acct_path) as f:
    for r in csv.DictReader(f):
        full = r['internal_account_number']
        internal_accounts.append({
            "name": r['internal_entity'],
            "institution": r.get('institution',''),
            "full": full,
            "last4": full[-4:]
        })



pconf = cfg['parser']
provider = pconf['provider']

print(f"[llm_parser] provider={provider!r}")
print(f"[llm_parser] loading HF branch" if provider=="hf" else "")

# Here we build a list of internal account names and numbers
# to help the LLM parse the email
# ── Build prompt with account mapping ─────────────────────────────────────
acct_lines = [
    f"- {a['name']} ({a['institution']}): acct={a['full']} (last4={a['last4']})"
    for a in internal_accounts
]
ACCT_REF = "Here are your internal accounts:\n" + "\n".join(acct_lines)

#print(ACCT_REF)
#exit()
# Prompt template
PROMPT = f"""

{ACCT_REF}


Extract the following fields as a JSON object (do not output any other text). Synonyms for the data that fills the fields are acceptable but the field names must be exactly as specified. Synonyms are given in square brackets. The fields are:
- date  (must extract and convert to ISO 8601 format (YYYY-MM-DD)
- internal_entity:string (name of the account_name or account holder)
- institution:string (name of the bank corresponding to internal entity)
- account_number:integer [account id]  (account number of the internal entity)
- external_entity:string [external account, external account name]
- amount:float (negative for debit to account number, positive for credit from account number)
- available_balance:float [balance] (available balance, as in remaining balance, in the account number)
- currency:string (base currency is South African Rand (ZAR))
- description:string (Nature of transaction, e.g. "ATM withdrawal", "POS purchase", "EFT payment" with description of reason for transaction if possible, e.g. "EFT payment for groceries")

You must ensure the following:
- Ensure that if there is no information for a field, you include that field in the JSON with an empty string.
- Ensure that you do not add any fields that are not in the list above.
- Enure that you extract the date from the email body first. If not available, then use metadata.
- Ensure that the amount is a number, not a string.
- Ensure that if you do not see information for a field, you do not include it in the JSON.
- Ensure that the JSON is valid and parsable.
- Ensure that you do not provide any other text in your output, literally only the JSON object. This includes any explanations, comments, notes, tips and your internal thought processes given step by step.
- Ensure that you do not add any fields that are not in the list above.
- Ensure that you do not use any abbreviations or short forms of words in your output.
- Ensure that you do not provide any other text in your output, literally only the JSON object. This includes any explanations, comments, notes, tips and your internal thought processes given step by step.

Here is the email metadata:
\"\"\"
{{metadata}}
\"\"\"
From: {{from_addr}}
Subject: {{subject}}

Here is the email body text:
\"\"\"
{{body}}
\"\"\"

JSON:
"""

# Initialize backends
if provider == "openai":
    import openai
    openai.api_key = keyring.get_password(
        cfg['openai']['keyring_service'], cfg['openai']['keyring_user']
    )
    if cfg.get('openai', {}).get('api_base'):
        openai.api_base = cfg['openai']['api_base']
    MODEL = pconf['openai_model']

    def generate(prompt):
        r = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You parse bank emails."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return r.choices[0].message.content

elif provider == "llamacpp":
    from llama_cpp import Llama
    llm = Llama(
        model_path=pconf['model_path'],
        n_ctx=pconf.get('n_ctx', 2048),
        seed=pconf.get('seed', 42)
    )

    def generate(prompt):
        r = llm(prompt, temperature=0.0, echo=False)
        return r['choices'][0]['text']

elif provider == "hf":
    import torch
    from transformers import (
        pipeline as hf_pipeline,
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    print("[llm_parser] Initializing HF backend…")

    # Quant config
    qb = pconf.get('quant_bits', 8)
    if qb == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=pconf.get('bnb_4bit_quant_type','nf4'),
            bnb_4bit_use_double_quant=pconf.get('bnb_double_quant',True),
            bnb_4bit_compute_dtype=getattr(torch, pconf['bnb_4bit_compute_dtype']),
        )
    elif qb == 8:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_cfg = None

    model_id = pconf.get('hf_model_id')
    if not model_id:
        raise ValueError("hf_model_id must be set in config under parser")

    # Device logic
    use_map = (pconf.get('device_map','auto') != 'none')
    device_map = 'auto' if use_map else None
    device     = pconf.get('device', -1)
    dtype      = torch.float16 if (device_map or device>=0) and torch.cuda.is_available() else torch.float32

    print(f"[llm_parser] Loading model {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "trust_remote_code": True,
        "max_new_tokens": pconf['max_new_tokens'],
        "do_sample": False,
        "return_full_text": False
    }
    if not use_map:
        print(f"[llm_parser] Pipeline on device {device}")
        pipe_kwargs["device"] = device

    llm_pipe = hf_pipeline("text-generation", **pipe_kwargs)
    print("[llm_parser] Huggingface pipeline ready")

    def generate(prompt):
        out = llm_pipe(prompt)
        return out[0]['generated_text']

else:
    raise ValueError(f"Unknown provider: {provider}")

class LLMParser(BankEmailParser):
    def parse(self, message: dict):
        # 1) Extract the plain-text body
        raw_body = ""
        for part in message['payload'].get('parts', []):
            if part.get('mimeType') == "text/plain":
                raw_body = part['body'].get('data',"")
                break
        if not raw_body:
            return []

        # 2) Decode & build prompt
        body   = base64.urlsafe_b64decode(raw_body).decode('utf-8', errors='ignore')
        print(f"[llm_parser] BODY >>> {body[:1000]}")

        # 2a) pull out From and Subject
        headers = { h['name']: h['value'] for h in message['payload']['headers'] }
        from_addr    = headers.get('From', '')
        subject      = headers.get('Subject', '')
        
        # 2b) optional “metadata” blob if you want to pass both together
        metadata_str = f"From: {from_addr}\nSubject: {subject}"
        prompt = PROMPT.format(metadata = metadata_str,
                                from_addr = from_addr,
                                subject   = subject,
                                body=body)

        # 3) Generate JSON text
        raw_json = generate(prompt).strip()
        print("[llm_parser] RAW_JSON >>>", repr(raw_json))   # debug
        print("\n\n\n")
        print("This is the genuine LLM ouput through the model as a json output")
        
        # 4) Ensure it ends with a brace
        if not raw_json.endswith("}"):
            raw_json += "}"

        def extract_first_json(raw: str) -> str | None:
            start = raw.find("{")
            if start < 0:
                return None
            depth = 0
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return raw[start : i+1]
            return None

        # 5) Parse JSON
        raw = generate(prompt).strip()
        # debug
        print("[llm_parser] RAW >>>", raw.replace("\n"," ") + "...")

        json_text = extract_first_json(raw)
        if not json_text:
            print("[llm_parser] no JSON found")
            return []
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return ["Error: JSONDecodeError"]
        

        # 6) Build Transaction
        try:
            dt = datetime.strptime(data['date'], "%Y-%m-%d")
        except:
            return ["Error: date parsing error"]


        print("Mama we made it")
        #exit()
        txn = Transaction(
            timestamp        = dt,
            account_name     = data.get('internal_entity', ""),
            institution     = data.get('institution', ""),
            account_number   = data.get('account_number', ""),
            external_entity  = data.get('external_entity', ""),
            amount           = float(data.get('amount', 0)),
            available_balance= float(data.get('available_balance', 0)),
            currency         = data.get('currency', ""),
            description      = data.get('description', ""),
            transaction_type = "unclassified",
            source_email     = next(
                (h['value'] for h in message['payload']['headers']
                 if h['name']=="From"), ""),
            email_id         = message.get('id', "")
        )
        return [txn]