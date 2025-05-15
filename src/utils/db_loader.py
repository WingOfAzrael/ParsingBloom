import yaml
import pandas as pd
from sqlalchemy import create_engine
from core.config_loader import load_config

def load_data_to_db():
    cfg = load_config(open("config/config.yaml"))['database']
    uri = (f"postgresql://{cfg['user']}@{cfg['host']}:{cfg['port']}/"
           f"{cfg['dbname']}")
    engine = create_engine(uri)

    df = pd.read_csv(cfg['paths']['master_csv'])
    df.to_sql(cfg['table'], engine, schema=cfg['schema'],
              if_exists='append', index=False)

if __name__ == "__main__":
    load_data_to_db()