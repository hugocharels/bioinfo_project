from numpy.lib import test
import pandas as pd
import torch
from torch.utils.data import Dataset


# Amino acid mapping for encoding
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # start from 1, 0 = padding

def encode_sequence(seq, max_len=1000):
    encoded = [AA_TO_INDEX.get(aa, 0) for aa in seq[:max_len]]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # pad to fixed length
    return encoded

class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_len=1000):
        self.data = dataframe.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = encode_sequence(row["Sequence"], self.max_len)
        labels = row.drop("Sequence").values.astype(float)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(labels, dtype=torch.float)


class ProteinDatasetLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.standard_columns = [
            "Cell membrane",
            "Cytoplasm",
            "Endoplasmic reticulum",
            "Golgi apparatus",
            "Lysosome/Vacuole",
            "Mitochondrion",
            "Nucleus",
            "Peroxisome",
            "Sequence",
        ]
        self.column_mapping = {"fasta": "Sequence", "Sequence": "Sequence"}

    def load(self):
        df = pd.read_csv(self.filepath)

        # Standardize column names
        df = self._standardize_columns(df)

        # Normalize label values to 0 or 1
        for col in self.standard_columns:
            if col != "Sequence":
                df[col] = df[col].fillna(0).apply(lambda x: int(float(x) > 0))

        # Keep only the standard columns in the right order
        df = df[[col for col in self.standard_columns if col in df.columns]]

        self.df = df
        return df

    def _standardize_columns(self, df):
        df = df.rename(columns=self.column_mapping)

        # Handle cases where labels are named differently or missing
        # e.g., Extracellular or Plastid may need to be dropped
        known_labels = set(self.standard_columns)
        df_labels = [col for col in df.columns if col in known_labels]

        # Drop unexpected columns
        df = df[df_labels + ["Sequence"]]

        # Fill missing label columns with 0s
        for label in known_labels:
            if label not in df.columns and label != "Sequence":
                df[label] = 0

        return df
