"""Loading pair tables. The competition data itself is never stored in this repo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEXT_COLUMNS = ("prompt", "response_a", "response_b")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read .parquet, .csv or .jsonl, with missing text fields as empty strings."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def a_won(df: pd.DataFrame) -> np.ndarray:
    """Boolean label from the competition's `winner` column ("model_a" / "model_b")."""
    labels = set(df["winner"].unique())
    if not labels <= {"model_a", "model_b"}:
        raise ValueError(f"unexpected winner labels: {sorted(labels - {'model_a', 'model_b'})}")
    return (df["winner"] == "model_a").to_numpy()
