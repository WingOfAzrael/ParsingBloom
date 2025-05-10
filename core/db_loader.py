"""
Graceful dual-loader:
• If Postgres details look valid and connection succeeds → upload.
• If Snowflake env-vars present and connect OK → upload.
Silent skips otherwise; never crashes the pipeline.
"""

import logging, os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from snowflake.connector import connect as sf_connect, errors as sf_errors

from config.config_loader import load_config

CFG  = load_config()

CSV     = CFG.paths.transactions_csv
DB      = CFG.database        # subscriptable / attribute
TYPE    = DB.type or "none"
TABLE   = DB.table
SCHEMA  = DB.db_schema        # <- renamed attribute

# --------------------------------------------------------------------------- #
def load_postgres():
    if TYPE != "postgres":
        return
    try:
        url = (f"postgresql+psycopg2://{DB['user']}@{DB['host']}:"
               f"{DB['port']}/{DB['dbname']}")
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute("select 1")
        df = pd.read_csv(CSV)
        df.to_sql(TABLE, url, schema=SCHEMA,
                  if_exists="append", index=False, method="multi")
        logging.info("Postgres: inserted %d rows.", len(df))
    except SQLAlchemyError as e:
        logging.warning("Postgres upload skipped: %s", e.__class__.__name__)

# --------------------------------------------------------------------------- #
def load_snowflake():
    env_req = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    ]
    if not all(k in os.environ for k in env_req):
        return
    try:
        df = pd.read_csv(CSV)
        ctx = sf_connect(
            account   = os.environ["SNOWFLAKE_ACCOUNT"],
            user      = os.environ["SNOWFLAKE_USER"],
            password  = os.environ["SNOWFLAKE_PASSWORD"],
            warehouse = os.environ["SNOWFLAKE_WAREHOUSE"],
            database  = os.environ["SNOWFLAKE_DATABASE"],
            schema    = os.environ["SNOWFLAKE_SCHEMA"],
            role      = os.getenv("SNOWFLAKE_ROLE"),
        )
        cs  = ctx.cursor()
        cs.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                date DATE, internal_account_number STRING,
                internal_entity STRING, institution STRING,
                external_entity STRING, amount FLOAT,
                available_balance FLOAT, currency STRING,
                description STRING, transaction_type STRING,
                source_email STRING, email_id STRING, run_id STRING
            )
        """)
        _, _, nrows, _ = cs.write_pandas(df, TABLE, quote_identifiers=False)
        logging.info("Snowflake: inserted %d rows.", nrows)
    except sf_errors.Error as e:
        logging.warning("Snowflake upload skipped: %s", e.__class__.__name__)
    finally:
        try:
            cs.close(); ctx.close()
        except Exception:
            pass
