import pandas as pd

def extract_csv(filepath: str):
  return pd.read_csv(filepath, header=None)